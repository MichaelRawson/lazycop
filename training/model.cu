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
#define THREADS 192

static const cudaDataType_t DATA_TYPE = CUDA_R_32F;
static const cublasLtOrder_t ORDER = CUBLASLT_ORDER_ROW;
static const cublasComputeType_t COMPUTE_TYPE = CUBLAS_COMPUTE_32F;

static cublasLtHandle_t BLAS_HANDLE;
static cublasLtMatmulDesc_t MM_DESC;
static cublasLtMatrixLayout_t WEIGHT_DESC;
static cublasLtMatmulPreference_t MM_PREFERENCE;
static float
	*EMBED_WEIGHTS = NULL,
	*OUT_WEIGHTS = NULL,
	*BACK_WEIGHTS = NULL,
	*BN_MEAN_WEIGHTS = NULL,
	*BN_VAR_WEIGHTS = NULL,
	*FC1_WEIGHTS = NULL,
	*FC1_BIAS = NULL,
	*FC2_WEIGHTS = NULL,
	*FC2_BIAS = NULL,
	*FC3_WEIGHTS = NULL;

static thread_local bool initialised = false;
static thread_local uint32_t
	num_nodes = 0,
	num_edges = 0,
	num_graphs = 0,
	*nodes = NULL,
	*sources = NULL,
	*targets = NULL,
	*batch = NULL,
	*counts = NULL;
static thread_local float
	*degree = NULL,
	*degree_t = NULL,
	*x = NULL,
	*out = NULL,
	*back = NULL,
	*save = NULL,
	*pooled = NULL,
	*fc1 = NULL,
	*fc2 = NULL,
	*fc3 = NULL;
static thread_local cublasLtMatrixLayout_t node_desc, pooled_desc;
static thread_local cublasLtMatmulHeuristicResult_t
	conv_heuristic,
	fc_heuristic;

static thread_local std::unordered_map<void *, uint32_t> cache;
template<typename T>
static void device_uninit(T **device, uint32_t count) {
	T* ptr = *device;
	if(ptr == NULL) {
		CUDAOK(cudaMalloc(device, count * sizeof(T)));
		cache[*device] = count;
		return;
	}
	uint32_t capacity = cache[ptr];
	if(capacity < count) {
		cache.erase(ptr);
		CUDAOK(cudaFree(ptr));
		CUDAOK(cudaMalloc(device, count * sizeof(T)));
		cache[*device] = count;
	}
}

template<typename T>
static void device_zeroed(T **device, uint32_t count) {
	device_uninit(device, count);
	CUDAOK(cudaMemsetAsync(
		*device,
		0,
		count * sizeof(T),
		CU_STREAM_PER_THREAD
	));
}

template<typename T>
static void device_memcpy(T *dst, T *src, uint32_t count) {
	CUDAOK(cudaMemcpyAsync(
		dst,
		src,
		count * sizeof(T),
		cudaMemcpyDeviceToDevice,
		CU_STREAM_PER_THREAD
	));
}

template<typename T>
static void upload_array(T **device, const T *data, uint32_t count) {
	device_uninit(device, count);
	CUDAOK(cudaMemcpyAsync(
		*device,
		data,
		count * sizeof(T),
		cudaMemcpyHostToDevice,
		CU_STREAM_PER_THREAD
	));
}

template<typename T>
static void download_array(T *dst, T *src, uint32_t count) {
	CUDAOK(cudaMemcpyAsync(
		dst,
		src,
		count * sizeof(T),
		cudaMemcpyDeviceToHost,
		CU_STREAM_PER_THREAD
	));
}

static void init_matrix_layout(
	cublasLtMatrixLayout_t layout,
	uint32_t rows,
	uint32_t cols
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
	uint32_t num_nodes,
	uint32_t num_edges,
	uint32_t *__restrict__ sources,
	uint32_t *__restrict__ targets,
	float *__restrict__ degree,
	float *__restrict__ degree_t
) {
	uint32_t thread = threadIdx.x;

	for(int i = thread; i < num_nodes; i += THREADS) {
		degree[i] = 1.0;
		degree_t[i] = 1.0;
	}
	__syncthreads();
	for(int i = thread; i < num_edges; i += THREADS) {
		atomicAdd(&degree[targets[i]], 1.0);
		atomicAdd(&degree_t[sources[i]], 1.0);
	}
	__syncthreads();
	for(int i = thread; i < num_nodes; i += THREADS) {
		degree[i] = 1.0 / degree[i];
		degree_t[i] = 1.0 / degree_t[i];
	}
}

static void compute_degree() {
	k_compute_degree<<<1, THREADS>>>(
		num_nodes,
		num_edges,
		sources,
		targets,
		degree,
		degree_t
	);
	CHECK_KERNEL;
}

__global__ void k_embed(
	uint32_t num_nodes,
	float *__restrict__ weights,
	uint32_t *__restrict__ nodes,
	float *__restrict__ x
) {
	uint32_t thread = threadIdx.x;
	uint32_t offset = thread / CHANNELS;
	uint32_t channel = thread % CHANNELS;

	for(int i = offset; i < num_nodes; i += (THREADS / CHANNELS)) {
		uint32_t flavour = nodes[i];
		float weight = weights[CHANNELS * flavour + channel];
		x[CHANNELS * i + channel] = weight;
	}
}

static void embed() {
	k_embed<<<1, THREADS>>>(num_nodes, EMBED_WEIGHTS, nodes, x);
	CHECK_KERNEL;
}

__global__ void k_sum_neighbours(
	uint32_t num_nodes,
	uint32_t num_edges,
	float *__restrict__ x,
	uint32_t *__restrict__ sources,
	uint32_t *__restrict__ targets,
	float *__restrict__ degree,
	float *__restrict__ degree_t,
	float *__restrict__ out,
	float *__restrict__ back
) {
	uint32_t thread = threadIdx.x;
	uint32_t offset = thread / CHANNELS;
	uint32_t channel = thread % CHANNELS;

	#pragma unroll 8
	for(int i = offset; i < num_nodes; i += (THREADS / CHANNELS)) {
		uint32_t index = CHANNELS * i + channel;
		out[index] = x[index] * degree[i];
		back[index] = x[index] * degree_t[i];
	}
	__syncthreads();

	#pragma unroll 8
	for(int i = offset; i < num_edges; i += (THREADS / CHANNELS)) {
		uint32_t source = CHANNELS * sources[i] + channel;
		uint32_t target = CHANNELS * targets[i] + channel;
		atomicAdd(&out[target], x[source] * degree[sources[i]]);
		atomicAdd(&back[source], x[target] * degree[targets[i]]);
	}
}

static void sum_neighbours() {
	k_sum_neighbours<<<1, THREADS>>>(
		num_nodes,
		num_edges,
		x,
		sources,
		targets,
		degree,
		degree_t,
		out,
		back
	);
	CHECK_KERNEL;
}

__global__ void k_preactivate(
	float *__restrict__ BN_MEAN_WEIGHTS,
	float *__restrict__ BN_VAR_WEIGHTS,
	float *__restrict__ x
) {
	uint32_t node = blockIdx.x;
	uint32_t channel = threadIdx.x;
	uint32_t index = CHANNELS * node + channel;
	float mean = BN_MEAN_WEIGHTS[channel];
	float variance = BN_VAR_WEIGHTS[channel];
	float numerator = max(0.0, x[index]) - mean;
	float denominator = __frsqrt_rn(variance);
	x[index] = numerator * denominator;
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
	uint32_t num_nodes,
	uint32_t num_graphs,
	uint32_t *__restrict__ batch,
	uint32_t *__restrict__ counts,
	float *__restrict__ x,
	float *__restrict__ pooled
) {
	uint32_t thread = threadIdx.x;
	uint32_t offset = thread / CHANNELS;
	uint32_t channel = thread % CHANNELS;

	for(int i = offset; i < num_nodes; i += (THREADS / CHANNELS)) {
		uint32_t graph = batch[i];
		uint32_t index = CHANNELS * i + channel;
		atomicAdd(&pooled[CHANNELS * graph + channel], x[index]);
	}
	__syncthreads();
	for(int i = offset; i < num_graphs; i += (THREADS / CHANNELS)) {
		uint32_t count = counts[i];
		uint32_t index = CHANNELS * i + channel;
		pooled[index] /= count;
	}
}

static void global_mean_pool() {
	k_global_mean_pool<<<1, THREADS>>>(
		num_nodes,
		num_graphs,
		batch,
		counts,
		x,
		pooled
	);
	CHECK_KERNEL;
}

__global__ void k_hidden_bias_relu(
	float *__restrict__ bias,
	float *__restrict__ x
) {
	uint32_t graph = blockIdx.x;
	uint32_t channel = threadIdx.x;
	uint32_t index = CHANNELS * graph + channel;
	x[index] = max(0.0, x[index] + bias[channel]);
}

static void hidden_bias_relu(float *bias, float *x) {
	k_hidden_bias_relu<<<num_graphs, CHANNELS>>>(bias, x);
	CHECK_KERNEL;
}

__global__ void k_output(
	float *__restrict__ weight,
	float *__restrict__ in,
	float *__restrict__ out
) {
	uint32_t graph = blockIdx.x;
	uint32_t channel = threadIdx.x;
	uint32_t index = CHANNELS * graph + channel;
	atomicAdd(
		&out[graph],
		weight[channel] * in[index] + (FC3_BIAS / CHANNELS)
	);
}

static void output(float *weight, float *in, float *out) {
	k_output<<<num_graphs, CHANNELS>>>(weight, in, out);
	CHECK_KERNEL;
}

static void mm(
	cublasLtMatmulHeuristicResult_t *heuristic,
	cublasLtMatrixLayout_t in_desc,
	float *in,
	cublasLtMatrixLayout_t weight_desc,
	float *weight,
	cublasLtMatrixLayout_t out_desc,
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
		out_desc,
		&heuristic->algo,
		NULL,
		0,
		CU_STREAM_PER_THREAD
	));
}

static void upload(
	const uint32_t *h_nodes,
	const uint32_t *h_sources,
	const uint32_t *h_targets,
	const uint32_t *h_batch,
	const uint32_t *h_counts,
	uint32_t h_num_nodes,
	uint32_t h_num_edges,
	uint32_t h_num_graphs
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
		BLASOK(cublasLtMatrixLayoutCreate(
			&pooled_desc,
			DATA_TYPE,
			0,
			0,
			0
		));
		initialised = true;
	}
	init_matrix_layout(node_desc, num_nodes, CHANNELS);
	init_matrix_layout(pooled_desc, num_graphs, CHANNELS);
	int num_results;
	BLASOK(cublasLtMatmulAlgoGetHeuristic(
		BLAS_HANDLE,
		MM_DESC,
		node_desc,
		WEIGHT_DESC,
		node_desc,
		node_desc,
		MM_PREFERENCE,
		1,
		&conv_heuristic,
		&num_results
	));
	BLASOK(cublasLtMatmulAlgoGetHeuristic(
		BLAS_HANDLE,
		MM_DESC,
		pooled_desc,
		WEIGHT_DESC,
		pooled_desc,
		pooled_desc,
		MM_PREFERENCE,
		1,
		&fc_heuristic,
		&num_results
	));

	device_uninit(&x, num_nodes * CHANNELS);
	device_uninit(&out, num_nodes * CHANNELS);
	device_uninit(&back, num_nodes * CHANNELS);
	device_uninit(&save, num_nodes * CHANNELS);
	device_zeroed(&pooled, num_graphs * CHANNELS);
	device_uninit(&fc1, num_graphs * CHANNELS);
	device_uninit(&fc2, num_graphs * CHANNELS);
	device_zeroed(&fc3, num_graphs);
	upload_array(&nodes, h_nodes, num_nodes);
	upload_array(&sources, h_sources, num_edges);
	upload_array(&targets, h_targets, num_edges);
	upload_array(&batch, h_batch, num_nodes);
	upload_array(&counts, h_counts, num_graphs);
	device_uninit(&degree, num_nodes);
	device_uninit(&degree_t, num_nodes);
	compute_degree();
}

static void conv0() {
	sum_neighbours();
	mm(
		&conv_heuristic,
		node_desc,
		out,
		WEIGHT_DESC,
		OUT_WEIGHTS,
		node_desc,
		x,
		false
	);
	mm(
		&conv_heuristic,
		node_desc,
		back,
		WEIGHT_DESC,
		BACK_WEIGHTS,
		node_desc,
		x,
		true
	);
}

static void residual(uint32_t layer) {
	uint32_t weight_offset;
	uint32_t bn_offset;

	device_memcpy(save, x, num_nodes * CHANNELS);
	bn_offset = CHANNELS * 2 * layer;
	preactivate(BN_MEAN_WEIGHTS + bn_offset, BN_VAR_WEIGHTS + bn_offset);
	sum_neighbours();
	weight_offset = CHANNELS * CHANNELS * (2 * layer + 1);
	mm(
		&conv_heuristic,
		node_desc,
		out,
		WEIGHT_DESC,
		OUT_WEIGHTS + weight_offset,
		node_desc,
		x,
		false
	);
	mm(
		&conv_heuristic,
		node_desc,
		back,
		WEIGHT_DESC,
		BACK_WEIGHTS + weight_offset,
		node_desc,
		x,
		true
	);

	bn_offset = CHANNELS * (2 * layer + 1);
	preactivate(BN_MEAN_WEIGHTS + bn_offset, BN_VAR_WEIGHTS + bn_offset);
	sum_neighbours();
	device_memcpy(x, save, num_nodes * CHANNELS);
	weight_offset = CHANNELS * CHANNELS * (2 * layer + 2);
	mm(
		&conv_heuristic,
		node_desc,
		out,
		WEIGHT_DESC,
		OUT_WEIGHTS + weight_offset,
		node_desc,
		x,
		true
	);
	mm(
		&conv_heuristic,
		node_desc,
		back,
		WEIGHT_DESC,
		BACK_WEIGHTS + weight_offset,
		node_desc,
		x,
		true
	);
}

extern "C" void init() {
	BLASOK(cublasLtCreate(&BLAS_HANDLE));
	BLASOK(cublasLtMatmulPreferenceCreate(&MM_PREFERENCE));
	BLASOK(cublasLtMatmulDescCreate(&MM_DESC, COMPUTE_TYPE, DATA_TYPE));
	BLASOK(cublasLtMatrixLayoutCreate(
		&WEIGHT_DESC,
		DATA_TYPE,
		0,
		0,
		0
	));
	init_matrix_layout(WEIGHT_DESC, CHANNELS, CHANNELS);

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
	upload_array(
		&FC1_WEIGHTS,
		FC1_WEIGHT_DATA,
		sizeof(FC1_WEIGHT_DATA) / sizeof(float)
	);
	upload_array(
		&FC1_BIAS,
		FC1_BIAS_DATA,
		sizeof(FC1_BIAS_DATA) / sizeof(float)
	);
	upload_array(
		&FC2_WEIGHTS,
		FC2_WEIGHT_DATA,
		sizeof(FC2_WEIGHT_DATA) / sizeof(float)
	);
	upload_array(
		&FC2_BIAS,
		FC2_BIAS_DATA,
		sizeof(FC2_BIAS_DATA) / sizeof(float)
	);
	upload_array(
		&FC3_WEIGHTS,
		FC3_WEIGHT_DATA,
		sizeof(FC3_WEIGHT_DATA) / sizeof(float)
	);
}

extern "C" void model(
	const uint32_t *h_nodes,
	const uint32_t *h_sources,
	const uint32_t *h_targets,
	const uint32_t *h_batch,
	const uint32_t *h_counts,
	uint32_t h_num_nodes,
	uint32_t h_num_edges,
	uint32_t h_num_graphs,
	float  *h_results
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
	mm(
		&fc_heuristic,
		pooled_desc,
		pooled,
		WEIGHT_DESC,
		FC1_WEIGHTS,
		pooled_desc,
		fc1,
		false
	);
	hidden_bias_relu(FC1_BIAS, fc1);
	mm(
		&fc_heuristic,
		pooled_desc,
		fc1,
		WEIGHT_DESC,
		FC2_WEIGHTS,
		pooled_desc,
		fc2,
		false
	);
	hidden_bias_relu(FC2_BIAS, fc2);
	output(FC3_WEIGHTS, fc2, fc3);
	download_array(h_results, fc3, num_graphs);
}

#ifndef NO_TEST
static void go() {
	const int num_nodes = 1000;
	const int num_edges = 2000;
	const int num_graphs = 10;
	uint32_t nodes[num_nodes];
	uint32_t sources[num_edges];
	uint32_t targets[num_edges];
	uint32_t batch[num_nodes];
	uint32_t counts[num_graphs];
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

	float results[num_graphs];
	model(
		nodes,
		sources,
		targets,
		batch,
		counts,
		num_nodes,
		num_edges,
		num_graphs,
		results
	);
	std::cout << results[0];
	for(int i = 1; i < num_graphs; i++) {
		std::cout << "," << results[i];
	}
	std::cout << std::endl;
}

int main() {
	init();

	std::vector<std::thread> workers;
	auto f = []() {
		for(int i = 0; i < 32; i++) {
			go();
		}
	};
	for(int i = 0; i < 64; i++) {
		workers.emplace_back(f);
	}
	for(auto &worker : workers) {
		worker.join();
	}
}
#endif
