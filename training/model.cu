#include <cassert>
#include <iostream>
#include <thread>
#include <vector>
#include <unordered_map>
#include <cuda.h>
#include <cublasLt.h>
#include "weights.h"

#define BLASOK(x) assert(x == CUBLAS_STATUS_SUCCESS)
#define CUDAOK(x) assert(x == cudaSuccess)
#define CHECK_KERNEL CUDAOK(cudaGetLastError())

static const cudaDataType_t DATA_TYPE = CUDA_R_32F;
static const cublasLtOrder_t ORDER = CUBLASLT_ORDER_ROW;
static const cublasComputeType_t COMPUTE_TYPE = CUBLAS_COMPUTE_32F;

static cublasLtHandle_t BLAS_HANDLE;
static cublasLtMatmulDesc_t MM_DESC;
static cublasLtMatrixLayout_t CONV_WEIGHT_DESC;
static cublasLtMatmulPreference_t MM_PREFERENCE;
static float
	*EMBED_WEIGHTS = NULL,
	*OUT_WEIGHTS = NULL,
	*BACK_WEIGHTS = NULL,
	*BN_MEAN_WEIGHTS = NULL,
	*BN_VAR_WEIGHTS = NULL;

static thread_local bool initialised = false;
static thread_local int32_t
	num_nodes = 0,
	num_edges = 0,
	num_graphs = 0,
	*nodes = NULL,
	*sources = NULL,
	*targets = NULL,
	*batch = NULL,
	*counts = NULL,
	*degree = NULL,
	*degree_t = NULL;
static thread_local float
	*x = NULL,
	*out = NULL,
	*back = NULL,
	*save = NULL,
	*pooled = NULL,
	*hidden = NULL;
static thread_local cublasLtMatrixLayout_t node_desc;
static thread_local cublasLtMatmulHeuristicResult_t conv_heuristic;

static thread_local std::unordered_map<void *, int32_t> cache;
template<typename T>
static void device_uninit(T **device, int32_t count) {
	T* ptr = *device;
	if(ptr == NULL) {
		CUDAOK(cudaMalloc(device, count * sizeof(T)));
		cache[*device] = count;
		return;
	}
	int32_t capacity = cache[ptr];
	if(capacity < count) {
		cache.erase(ptr);
		CUDAOK(cudaFree(ptr));
		CUDAOK(cudaMalloc(device, count * sizeof(T)));
		cache[*device] = count;
	}
}

template<typename T>
static void device_zeroed(T **device, int32_t count) {
	device_uninit(device, count);
	CUDAOK(cudaMemsetAsync(*device, 0, count * sizeof(T)));
}

template<typename T>
static void device_memcpy(T *dst, T *src, int32_t count) {
	CUDAOK(cudaMemcpyAsync(
		dst,
		src,
		count * sizeof(T),
		cudaMemcpyDeviceToDevice
	));
}

template<typename T>
static void upload_array(T **device, const T *data, int32_t count) {
	device_uninit(device, count);
	CUDAOK(cudaMemcpyAsync(
		*device,
		data,
		count * sizeof(T),
		cudaMemcpyHostToDevice
	));
}

template<typename T>
static T download_scalar(T *device) {
	float result;
	CUDAOK(cudaMemcpy(
		&result,
		device,
		sizeof(float),
		cudaMemcpyDeviceToHost
	));
	return result;
}

static void init_matrix_layout(
	cublasLtMatrixLayout_t layout,
	int32_t rows,
	int32_t cols
) {
	BLASOK(cublasLtMatrixLayoutInit(
		layout,
		DATA_TYPE,
		rows,
		cols,
		cols
	));
	BLASOK(cublasLtMatrixLayoutSetAttribute(
		layout,
		CUBLASLT_MATRIX_LAYOUT_ORDER,
		&ORDER,
		sizeof(ORDER)
	));
}

__global__ void k_compute_degree(
	int32_t *__restrict__ sources,
	int32_t *__restrict__ targets,
	int32_t *__restrict__ degree,
	int32_t *__restrict__ degree_t
) {
	int32_t edge = blockIdx.x;
	atomicAdd(&degree[targets[edge]], 1);
	atomicAdd(&degree_t[sources[edge]], 1);
}

static void compute_degree() {
	k_compute_degree<<<num_edges, 1>>>(
		sources,
		targets,
		degree,
		degree_t
	);
	CHECK_KERNEL;
}

__global__ void k_embed(
	float *__restrict__ weights,
	int32_t *__restrict__ nodes,
	float *__restrict__ x
) {
	int32_t node = blockIdx.x;
	int32_t channel = threadIdx.x;
	int32_t flavour = nodes[node];
	x[CHANNELS * node + channel] = weights[CHANNELS * flavour + channel];
}

static void embed() {
	k_embed<<<num_nodes, CHANNELS>>>(EMBED_WEIGHTS, nodes, x);
	CHECK_KERNEL;
}

__global__ void k_sum_neighbours_normalise(
	int32_t num_nodes,
	int32_t num_edges,
	float *__restrict__ x,
	int32_t *__restrict__ sources,
	int32_t *__restrict__ targets,
	float *__restrict__ out,
	float *__restrict__ back,
	int32_t *__restrict__ degree,
	int32_t *__restrict__ degree_t
) {
	int32_t channel = threadIdx.x;
	#pragma unroll
	for(int i = 0; i < num_nodes; i++) {
		int32_t index = CHANNELS * i + channel;
		out[index] = x[index];
		back[index] = x[index];
	}
	#pragma unroll
	for(int i = 0; i < num_edges; i++) {
		int32_t source = sources[i];
		int32_t target = targets[i];
		int32_t source_index = source * CHANNELS + channel;
		int32_t target_index = target * CHANNELS + channel;
		out[target_index] += x[source_index];
		back[source_index] += x[target_index];
	}
	#pragma unroll
	for(int i = 0; i < num_nodes; i++) {
		int32_t index = CHANNELS * i + channel;
		out[index] /= degree[i] + 1;
		back[index] /= degree_t[i] + 1;
	}
}

static void sum_neighbours_normalise() {
	k_sum_neighbours_normalise<<<1, CHANNELS>>>(
		num_nodes,
		num_edges,
		x,
		sources,
		targets,
		out,
		back,
		degree,
		degree_t
	);
	CHECK_KERNEL;
}

/*
__global__ void k_normalise(
	int32_t num_nodes,
	float *__restrict__ out,
	float *__restrict__ back,
	int32_t *__restrict__ degree,
	int32_t *__restrict__ degree_t
) {
	int32_t channel = threadIdx.x;
	for(int i = 0; i < num_nodes; i++) {
		int32_t index = CHANNELS * i + channel;
		out[index] /= degree[i] + 1;
		back[index] /= degree_t[i] + 1;
	}
}

static void normalise() {
	k_normalise<<<1, CHANNELS>>>(
		num_nodes,
		out,
		back,
		degree,
		degree_t
	);
	CHECK_KERNEL;
}
*/

__global__ void k_preactivate(
	float *__restrict__ BN_MEAN_WEIGHTS,
	float *__restrict__ BN_VAR_WEIGHTS,
	float *__restrict__ x
) {
	int32_t node = blockIdx.x;
	int32_t channel = threadIdx.x;
	int32_t index = CHANNELS * node + channel;
	float mean = BN_MEAN_WEIGHTS[channel];
	float var = BN_VAR_WEIGHTS[channel];
	x[index] = (max(0.0, x[index]) - mean) / sqrt(var + BN_EPS);
}

static void preactivate(float *BN_MEAN_WEIGHTS, float *BN_VAR_WEIGHTS) {
	k_preactivate<<<num_nodes, CHANNELS>>>(
		BN_MEAN_WEIGHTS,
		BN_VAR_WEIGHTS,
		x
	);
	CHECK_KERNEL;
}

__global__ void k_global_mean_pool(
	int32_t *__restrict__ batch,
	int32_t *__restrict__ counts,
	float *__restrict__ x,
	float *__restrict__ pooled
) {
	int32_t node = blockIdx.x;
	int32_t channel = threadIdx.x;
	int32_t graph = batch[node];
	int32_t count = counts[graph];
	int32_t index = CHANNELS * node + channel;
	atomicAdd(
		&pooled[CHANNELS * graph + channel],
		x[index] / count
	);
}

static void global_mean_pool() {
	k_global_mean_pool<<<num_nodes, CHANNELS>>>(
		batch,
		counts,
		x,
		pooled
	);
	CHECK_KERNEL;
}

static void mm(
	cublasLtMatmulAlgo_t *algorithm,
	cublasLtMatrixLayout_t in_desc,
	float *in,
	cublasLtMatrixLayout_t weight_desc,
	float *weight,
	float *out,
	bool accumulate
) {
	float alpha = 1.0;
	float beta = accumulate ? 1.0 : 0.0;
	BLASOK(cublasLtMatmul(
		BLAS_HANDLE,
		MM_DESC,
		&alpha,
		in,
		in_desc,
		weight,
		weight_desc,
		&beta,
		out,
		in_desc,
		out,
		in_desc,
		algorithm,
		NULL,
		0,
		CU_STREAM_PER_THREAD
	));
}

static void upload(
	const int32_t *h_nodes,
	const int32_t *h_sources,
	const int32_t *h_targets,
	const int32_t *h_batch,
	const int32_t *h_counts,
	int32_t h_num_nodes,
	int32_t h_num_edges,
	int32_t h_num_graphs
) {
	num_nodes = h_num_nodes;
	num_edges = h_num_edges;
	num_graphs = h_num_graphs;

	if(!initialised) {
		BLASOK(cublasLtMatrixLayoutCreate(
			&node_desc,
			DATA_TYPE,
			0,
			0,
			0
		));
		initialised = true;
	}
	init_matrix_layout(node_desc, num_nodes, CHANNELS);
	int32_t num_results;
	BLASOK(cublasLtMatmulAlgoGetHeuristic(
		BLAS_HANDLE,
		MM_DESC,
		node_desc,
		CONV_WEIGHT_DESC,
		node_desc,
		node_desc,
		MM_PREFERENCE,
		1,
		&conv_heuristic,
		&num_results
	));

	device_uninit(&x, num_nodes * CHANNELS);
	device_uninit(&out, num_nodes * CHANNELS);
	device_uninit(&back, num_nodes * CHANNELS);
	device_uninit(&save, num_nodes * CHANNELS);
	device_zeroed(&pooled, num_graphs * CHANNELS);
	device_uninit(&hidden, num_graphs * HIDDEN);
	device_zeroed(&degree, num_nodes);
	device_zeroed(&degree_t, num_nodes);
	upload_array(&nodes, h_nodes, num_nodes);
	upload_array(&sources, h_sources, num_edges);
	upload_array(&targets, h_targets, num_edges);
	upload_array(&batch, h_batch, num_nodes);
	upload_array(&counts, h_counts, num_graphs);
	compute_degree();
}

static void conv0() {
	sum_neighbours_normalise();
	mm(
		&conv_heuristic.algo,
		node_desc,
		out,
		CONV_WEIGHT_DESC,
		OUT_WEIGHTS,
		x,
		false
	);
	mm(
		&conv_heuristic.algo,
		node_desc,
		back,
		CONV_WEIGHT_DESC,
		BACK_WEIGHTS,
		x,
		true
	);
}

static void residual(int32_t layer) {
	int32_t weight_offset;
	int32_t bn_offset;

	device_memcpy(save, x, num_nodes * CHANNELS);
	bn_offset = CHANNELS * 2 * layer;
	preactivate(BN_MEAN_WEIGHTS + bn_offset, BN_VAR_WEIGHTS + bn_offset);
	sum_neighbours_normalise();
	weight_offset = CHANNELS * CHANNELS * (2 * layer + 1);
	mm(
		&conv_heuristic.algo,
		node_desc,
		out,
		CONV_WEIGHT_DESC,
		OUT_WEIGHTS + weight_offset,
		x,
		false
	);
	mm(
		&conv_heuristic.algo,
		node_desc,
		back,
		CONV_WEIGHT_DESC,
		BACK_WEIGHTS + weight_offset,
		x,
		true
	);

	bn_offset = CHANNELS * (2 * layer + 1);
	preactivate(BN_MEAN_WEIGHTS + bn_offset, BN_VAR_WEIGHTS + bn_offset);
	sum_neighbours_normalise();
	device_memcpy(x, save, num_nodes * CHANNELS);
	weight_offset = CHANNELS * CHANNELS * (2 * layer + 2);
	mm(
		&conv_heuristic.algo,
		node_desc,
		out,
		CONV_WEIGHT_DESC,
		OUT_WEIGHTS + weight_offset,
		x,
		true
	);
	mm(
		&conv_heuristic.algo,
		node_desc,
		back,
		CONV_WEIGHT_DESC,
		BACK_WEIGHTS + weight_offset,
		x,
		true
	);
}

extern "C" void init() {
	BLASOK(cublasLtCreate(&BLAS_HANDLE));
	BLASOK(cublasLtMatmulPreferenceCreate(&MM_PREFERENCE));
	BLASOK(cublasLtMatmulDescCreate(&MM_DESC, COMPUTE_TYPE, DATA_TYPE));
	BLASOK(cublasLtMatrixLayoutCreate(
		&CONV_WEIGHT_DESC,
		DATA_TYPE,
		0,
		0,
		0
	));
	init_matrix_layout(CONV_WEIGHT_DESC, CHANNELS, CHANNELS);

	upload_array(
		&EMBED_WEIGHTS,
		EMBED_WEIGHTS_DATA,
		sizeof(EMBED_WEIGHTS_DATA) / sizeof(float)
	);
	upload_array(
		&OUT_WEIGHTS,
		OUT_WEIGHTS_DATA,
		sizeof(OUT_WEIGHTS_DATA) / sizeof(float)
	);
	upload_array(
		&BACK_WEIGHTS,
		BACK_WEIGHTS_DATA,
		sizeof(BACK_WEIGHTS_DATA) / sizeof(float)
	);
	upload_array(
		&BN_MEAN_WEIGHTS,
		BN_MEAN_WEIGHTS_DATA,
		sizeof(BN_MEAN_WEIGHTS_DATA) / sizeof(float)
	);
	upload_array(
		&BN_VAR_WEIGHTS,
		BN_VAR_WEIGHTS_DATA,
		sizeof(BN_VAR_WEIGHTS_DATA) / sizeof(float)
	);
}

extern "C" float model(
	const int32_t *h_nodes,
	const int32_t *h_sources,
	const int32_t *h_targets,
	const int32_t *h_batch,
	const int32_t *h_counts,
	int32_t h_num_nodes,
	int32_t h_num_edges,
	int32_t h_num_graphs
) {
	upload(
		h_nodes,
		h_sources,
		h_targets,
		h_batch,
		h_counts,
		h_num_nodes,
		h_num_edges,
		h_num_graphs
	);
	embed();
	conv0();
	for(int i = 0; i < MODULES; i++) {
		residual(i);
	}
	global_mean_pool();
	return download_scalar(x);
}

#ifndef NO_TEST
static void go() {
	const int num_nodes = 1000;
	const int num_edges = 2000;
	const int num_graphs = 100;
	int32_t nodes[num_nodes];
	int32_t sources[num_edges];
	int32_t targets[num_edges];
	int32_t batch[num_nodes];
	int32_t counts[num_graphs];
	for(int i = 0; i < num_nodes; i++) {
		nodes[i] = 0;
		batch[i] = i / (num_nodes / num_graphs);
	}
	for(int i = 0; i < num_edges; i++) {
		sources[i] = i / 2;
		targets[i] = i / 2;
	}
	for(int i = 0; i < num_graphs; i++) {
		counts[i] = num_nodes / num_graphs;
	}
	float result = model(
		nodes,
		sources,
		targets,
		batch,
		counts,
		num_nodes,
		num_edges,
		num_graphs
	);
	std::cout << result << std::endl;
}

int main() {
	init();

	std::vector<std::thread> workers;
	auto f = []() {
		for(int i = 0; i < 100; i++) {
			go();
		}
	};
	//f();
	for(int i = 0; i < 8; i++) {
		workers.emplace_back(f);
	}
	for(auto &worker : workers) {
		worker.join();
	}
}
#endif
