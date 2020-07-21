#include <cassert>
#include <cuda.h>
#include <cublasLt.h>
#include "weights.h"

static const size_t
	ALIGNTO = 256,
	THREADS = 512,
	UNROLL = 8;
static const cudaDataType_t DATA_TYPE = CUDA_R_32F;
static const cublasComputeType_t COMPUTE_TYPE = CUBLAS_COMPUTE_32F;

#define BLASOK(x) assert(x == CUBLAS_STATUS_SUCCESS)
#define CUDAOK(x) assert(x == cudaSuccess)
#define ALIGN(x) (ALIGNTO * (((x) / ALIGNTO) + ((x) % ALIGNTO != 0)))

#define DEVICE_ALLOC(device, count) {\
	if(device ## _capacity < count) {\
		CUDAOK(cudaFree(device));\
		CUDAOK(cudaMalloc(&device, count * sizeof(*device)));\
		device ## _capacity = count;\
	}\
}

#define PAGE_ALLOC(page, count) {\
	if(page ## _capacity < count) {\
		CUDAOK(cudaFreeHost(page));\
		CUDAOK(cudaMallocHost(&page, count * sizeof(*page)));\
		page ## _capacity = count;\
	}\
}

// global variables, not changed after init()
static cublasLtHandle_t BLAS_HANDLE;
static cublasLtMatmulDesc_t MM_DESC;
static cublasLtMatrixLayout_t
	CONV_WEIGHT_DESC,
	HIDDEN_WEIGHT_DESC,
	OUTPUT_WEIGHT_DESC;
static cublasLtMatmulPreference_t MM_PREFERENCE;
static float
	*EMBED_WEIGHTS = NULL,
	*OUT_WEIGHTS = NULL,
	*BACK_WEIGHTS = NULL,
	*HIDDEN_WEIGHTS = NULL,
	*HIDDEN_BIAS = NULL,
	*OUTPUT_WEIGHTS = NULL;

// thread-local global variables, changed once-per-cycle
static thread_local bool thread_initialised = false;
static thread_local uint8_t *p_upload, *d_upload;
static thread_local float *p_download, *d_download;
static thread_local int32_t
	*nodes = NULL,
	*node_batch = NULL,
	*edge_batch = NULL;
static thread_local int2 *edges = NULL;
static thread_local float
	*forward_node_norm = NULL,
	*backward_node_norm = NULL,
	*x = NULL,
	*out = NULL,
	*back = NULL,
	*scratch1 = NULL,
	*scratch2 = NULL,
	*pooled = NULL,
	*hidden = NULL;
static thread_local uint32_t
	num_nodes = 0,
	num_edges = 0,
	num_graphs = 0,
	p_upload_capacity = 0,
	d_upload_capacity = 0,
	p_download_capacity = 0,
	d_download_capacity = 0,
	x_capacity = 0,
	out_capacity = 0,
	back_capacity = 0,
	scratch1_capacity = 0,
	scratch2_capacity = 0,
	pooled_capacity = 0,
	hidden_capacity = 0;
static thread_local cublasLtMatrixLayout_t
	node_desc,
	pooled_desc,
	hidden_desc,
	output_desc;
static thread_local cublasLtMatmulHeuristicResult_t
	conv_heuristic,
	hidden_heuristic,
	output_heuristic;

static void upload_weights(float **device, const float *data, size_t size) {
	CUDAOK(cudaMalloc(device, size));
	CUDAOK(cudaMemcpy(
		*device,
		data,
		size,
		cudaMemcpyHostToDevice
	));
}

static void init_matrix_layout(
	cublasLtMatrixLayout_t layout,
	int32_t rows,
	int32_t cols
) {
	cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
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
		&order,
		sizeof(order)
	));
}

static void mm(
	cublasLtMatmulHeuristicResult_t *heuristic,
	cublasLtMatrixLayout_t a_desc,
	float *a,
	cublasLtMatrixLayout_t b_desc,
	float *b,
	cublasLtMatrixLayout_t out_desc,
	float *out
) {
	float alpha = 1.0f;
	float beta = 0.0f;
	BLASOK(cublasLtMatmul(
		BLAS_HANDLE,
		MM_DESC,
		&alpha,
		a,
		a_desc,
		b,
		b_desc,
		&beta,
		out,
		out_desc,
		out,
		out_desc,
		&heuristic->algo,
		NULL,
		0,
		CU_STREAM_PER_THREAD
	));
}

__global__ void k_embed(
	int32_t num_nodes,
	int32_t *nodes,
	float *weights,
	float *x
) {
	auto thread = threadIdx.x;
	auto channel = thread % CHANNELS;
	auto offset = thread / CHANNELS;

	#pragma unroll UNROLL
	for(int i = offset; i < num_nodes; i += THREADS / CHANNELS) {
		auto index = CHANNELS * i + channel;
		auto node = __ldcs(nodes + i);
		auto weight = __ldg(weights + CHANNELS * node + channel);
		__stcg(x + index, weight);
	}
}

static void embed() {
	k_embed<<<1, THREADS>>>(
		num_nodes,
		nodes,
		EMBED_WEIGHTS,
		x
	);
}

__global__ void k_gather_neighbours(
	uint32_t num_edges,
	float *x,
	int2 *edges,
	int32_t *edge_batch,
	float *out,
	float *back
) {
	auto graph = blockIdx.x;
	auto channel = threadIdx.x;

	auto start = __ldg(edge_batch + graph);
	auto finish = __ldg(edge_batch + graph + 1);
	#pragma unroll UNROLL
	for(int i = start; i < finish; i++) {
		auto edge = __ldg(edges + i);
		auto from = CHANNELS * edge.x + channel;
		auto to = CHANNELS * edge.y + channel;
		atomicAdd(back + from, __ldg(x + to));
		atomicAdd(out + to, __ldg(x + from));
	}
}

static void gather_neighbours() {
	CUDAOK(cudaMemcpyAsync(
		out,
		x,
		num_nodes * CHANNELS * sizeof(float),
		cudaMemcpyDeviceToDevice,
		CU_STREAM_PER_THREAD
	));
	CUDAOK(cudaMemcpyAsync(
		back,
		x,
		num_nodes * CHANNELS * sizeof(float),
		cudaMemcpyDeviceToDevice,
		CU_STREAM_PER_THREAD
	));
	k_gather_neighbours<<<num_graphs, CHANNELS>>>(
		num_edges,
		x,
		edges,
		edge_batch,
		out,
		back
	);
}

__global__ void k_normalise(
	int32_t num_nodes,
	float *forward_node_norm,
	float *backward_node_norm,
	float *out,
	float *back
) {
	auto thread = threadIdx.x;
	auto channel = thread % CHANNELS;
	auto offset = thread / CHANNELS;

	#pragma unroll UNROLL
	for(int i = offset; i < num_nodes; i += THREADS / CHANNELS) {
		auto index = CHANNELS * i + channel;
		float forward_norm = __ldg(forward_node_norm + i);
		float backward_norm = __ldg(backward_node_norm + i);
		float out_val = forward_norm * __ldcs(out + index);
		float back_val = backward_norm * __ldcs(back + index);
		__stcg(out + index, out_val);
		__stcg(back + index, back_val);
	}
}

static void normalise() {
	k_normalise<<<1, THREADS>>>(
		num_nodes,
		forward_node_norm,
		backward_node_norm,
		out,
		back
	);
}

__global__ void k_combine_scratch(
	int32_t num_nodes,
	float *scratch1,
	float *scratch2,
	float *x
) {
	auto thread = threadIdx.x;
	#pragma unroll UNROLL
	for(int i = thread; i < CHANNELS * num_nodes; i += THREADS) {
		float current = __ldg(x + i);
		float combined = current +
			fmaxf(0.0f, __ldg(scratch1 + i)) +
			fmaxf(0.0f, __ldg(scratch2 + i));
		__stcg(x + i, combined);
	}
}

static void combine_scratch() {
	k_combine_scratch<<<1, THREADS>>>(
		num_nodes,
		scratch1,
		scratch2,
		x
	);
}

__global__ void k_global_mean_pool(
	int32_t num_nodes,
	int32_t num_graphs,
	int32_t *node_batch,
	float *x,
	float *pooled
) {
	auto graph = blockIdx.x;
	auto channel = threadIdx.x;

	auto start = __ldg(node_batch + graph);
	auto finish = __ldg(node_batch + graph + 1);
	auto count = finish - start;
	float norm = 1.0 / count;
	float *addr = pooled + CHANNELS * graph + channel;

	float total = 0.0;
	#pragma unroll UNROLL
	for(int i = start; i < finish; i++) {
		total += __ldg(x + CHANNELS * i + channel);
	}
	__stcg(addr, norm * total);
}

static void global_mean_pool() {
	k_global_mean_pool<<<num_graphs, CHANNELS>>>(
		num_nodes,
		num_graphs,
		node_batch,
		x,
		pooled
	);
}

__global__ void k_hidden_bias_relu(
	int32_t num_graphs,
	float *bias,
	float *hidden
) {
	auto channel = threadIdx.x;
	#pragma unroll UNROLL
	for(int i = 0; i < num_graphs; i++) {
		auto index = HIDDEN * i + channel;
		float activated = fmaxf(
			0.0f,
			__ldcs(hidden + index) + __ldg(bias + channel)
		);
		__stcg(hidden + index, activated);
	}
}

static void hidden_bias_relu() {
	k_hidden_bias_relu<<<1, HIDDEN>>>(num_graphs, HIDDEN_BIAS, hidden);
}

static void residual(int32_t layer) {
	float *out_weights = OUT_WEIGHTS + CHANNELS * CHANNELS * layer;
	float *back_weights = BACK_WEIGHTS + CHANNELS * CHANNELS * layer;

	gather_neighbours();
	normalise();
	mm(
		&conv_heuristic, 
		node_desc,
		out,
		CONV_WEIGHT_DESC,
		out_weights,
		node_desc,
		scratch1
	);
	mm(
		&conv_heuristic, 
		node_desc,
		back,
		CONV_WEIGHT_DESC,
		back_weights,
		node_desc,
		scratch2
	);
	combine_scratch();
}

static void upload(
	uint32_t h_num_nodes,
	uint32_t h_num_edges,
	uint32_t h_num_graphs,
	const uint32_t *h_nodes,
	const uint32_t *h_sources,
	const uint32_t *h_targets,
	const uint32_t *h_node_batch,
	const uint32_t *h_edge_batch
) {
	num_nodes = h_num_nodes;
	num_edges = h_num_edges;
	num_graphs = h_num_graphs;

	if(!thread_initialised) {
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
		BLASOK(cublasLtMatrixLayoutCreate(
			&hidden_desc,
			DATA_TYPE,
			0,
			0,
			0
		));
		BLASOK(cublasLtMatrixLayoutCreate(
			&output_desc,
			DATA_TYPE,
			0,
			0,
			0
		));
		thread_initialised = true;
	}
	init_matrix_layout(node_desc, num_nodes, CHANNELS);
	init_matrix_layout(pooled_desc, num_graphs, CHANNELS);
	init_matrix_layout(hidden_desc, num_graphs, 1024);
	init_matrix_layout(output_desc, num_graphs, 1);
	int _num_results;
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
		&_num_results
	));
	BLASOK(cublasLtMatmulAlgoGetHeuristic(
		BLAS_HANDLE,
		MM_DESC,
		pooled_desc,
		HIDDEN_WEIGHT_DESC,
		hidden_desc,
		hidden_desc,
		MM_PREFERENCE,
		1,
		&hidden_heuristic,
		&_num_results
	));
	BLASOK(cublasLtMatmulAlgoGetHeuristic(
		BLAS_HANDLE,
		MM_DESC,
		hidden_desc,
		OUTPUT_WEIGHT_DESC,
		output_desc,
		output_desc,
		MM_PREFERENCE,
		1,
		&output_heuristic,
		&_num_results
	));

	size_t node_offset = 0;
	auto edge_offset = ALIGN(
		node_offset + num_nodes * sizeof(int32_t)
	);
	auto forward_node_norm_offset = ALIGN(
		edge_offset + num_edges * sizeof(int2)
	);
	auto backward_node_norm_offset = ALIGN(
		forward_node_norm_offset + num_nodes * sizeof(float)
	);
	auto node_batch_offset = ALIGN(
		backward_node_norm_offset + num_nodes * sizeof(float)
	);
	auto edge_batch_offset = ALIGN(
		node_batch_offset + (num_graphs + 1) * sizeof(int32_t)
	);
	auto upload_bytes = edge_batch_offset +
		(num_graphs + 1) * sizeof(int32_t);

	DEVICE_ALLOC(d_upload, upload_bytes);
	DEVICE_ALLOC(d_download, num_graphs);
	DEVICE_ALLOC(x, num_nodes * CHANNELS);
	DEVICE_ALLOC(out, num_nodes * CHANNELS);
	DEVICE_ALLOC(back, num_nodes * CHANNELS);
	DEVICE_ALLOC(scratch1, num_nodes * CHANNELS);
	DEVICE_ALLOC(scratch2, num_nodes * CHANNELS);
	DEVICE_ALLOC(pooled, num_graphs * CHANNELS);
	DEVICE_ALLOC(hidden, num_graphs * HIDDEN);
	PAGE_ALLOC(p_upload, upload_bytes);
	PAGE_ALLOC(p_download, num_graphs);

	nodes = (int32_t *)(d_upload + node_offset);
	edges = (int2 *)(d_upload + edge_offset);
	forward_node_norm = (float *)(d_upload + forward_node_norm_offset);
	backward_node_norm = (float *)(d_upload + backward_node_norm_offset);
	node_batch = (int32_t *)(d_upload + node_batch_offset);
	edge_batch = (int32_t *)(d_upload + edge_batch_offset);

	// alignment-safe: should be aligned from before
	auto p_nodes = (int32_t *)(p_upload + node_offset);
	auto p_edges = (int2 *)(p_upload + edge_offset);
	auto p_forward_node_norm =
		(float *)(p_upload + forward_node_norm_offset);
	auto p_backward_node_norm =
		(float *)(p_upload + backward_node_norm_offset);
	auto p_node_batch = (int32_t *)(p_upload + node_batch_offset);
	auto p_edge_batch = (int32_t *)(p_upload + edge_batch_offset);

	memset(p_forward_node_norm, 0, num_nodes * sizeof(float));
	memset(p_backward_node_norm, 0, num_nodes * sizeof(float));
	for(uint32_t i = 0; i < num_edges; i++) {
		auto source = h_sources[i];
		auto target = h_targets[i];
		p_edges[i].x = source;
		p_edges[i].y = target;
		p_forward_node_norm[target] += 1.0f;
		p_backward_node_norm[source] += 1.0f;
	}
	for(uint32_t i = 0; i < num_nodes; i++) {
		p_nodes[i] = h_nodes[i];
		p_forward_node_norm[i] = 1.0f /
			(1.0f + p_forward_node_norm[i]);
		p_backward_node_norm[i] = 1.0f /
			(1.0f + p_backward_node_norm[i]);
	}
	p_node_batch[0] = 0;
	p_edge_batch[0] = 0;
	for(uint32_t i = 1; i <= num_graphs; i++) {
		p_node_batch[i] = h_node_batch[i - 1];
		p_edge_batch[i] = h_edge_batch[i - 1];
	}

	CUDAOK(cudaMemcpyAsync(
		d_upload,
		p_upload,
		upload_bytes,
		cudaMemcpyHostToDevice,
		CU_STREAM_PER_THREAD
	));
}

static void download(float *h_results) {
	CUDAOK(cudaMemcpyAsync(
		p_download,
		d_download,
		num_graphs * sizeof(float),
		cudaMemcpyDeviceToHost,
		CU_STREAM_PER_THREAD
	));
	CUDAOK(cudaStreamSynchronize(CU_STREAM_PER_THREAD));

	for(uint32_t i = 0; i < num_graphs; i++) {
		h_results[i] = p_download[i] + OUTPUT_BIAS;
	}
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
	BLASOK(cublasLtMatrixLayoutCreate(
		&HIDDEN_WEIGHT_DESC,
		DATA_TYPE,
		0,
		0,
		0
	));
	init_matrix_layout(HIDDEN_WEIGHT_DESC, CHANNELS, HIDDEN);
	BLASOK(cublasLtMatrixLayoutCreate(
		&OUTPUT_WEIGHT_DESC,
		DATA_TYPE,
		0,
		0,
		0
	));
	init_matrix_layout(OUTPUT_WEIGHT_DESC, HIDDEN, 1);

	upload_weights(
		&EMBED_WEIGHTS,
		EMBED_WEIGHTS_DATA,
		sizeof(EMBED_WEIGHTS_DATA)
	);
	upload_weights(
		&OUT_WEIGHTS,
		OUT_WEIGHTS_DATA,
		sizeof(OUT_WEIGHTS_DATA)
	);
	upload_weights(
		&BACK_WEIGHTS,
		BACK_WEIGHTS_DATA,
		sizeof(BACK_WEIGHTS_DATA)
	);
	upload_weights(
		&HIDDEN_WEIGHTS,
		HIDDEN_WEIGHT_DATA,
		sizeof(HIDDEN_WEIGHT_DATA)
	);
	upload_weights(
		&HIDDEN_BIAS,
		HIDDEN_BIAS_DATA,
		sizeof(HIDDEN_BIAS_DATA)
	);
	upload_weights(
		&OUTPUT_WEIGHTS,
		OUTPUT_WEIGHT_DATA,
		sizeof(OUTPUT_WEIGHT_DATA)
	);
}

extern "C" void model(
	uint32_t h_num_nodes,
	uint32_t h_num_edges,
	uint32_t h_num_graphs,
	const uint32_t *h_nodes,
	const uint32_t *h_sources,
	const uint32_t *h_targets,
	const uint32_t *h_node_batch,
	const uint32_t *h_edge_batch,
	float *h_results
) {
	upload(
		h_num_nodes,
		h_num_edges,
		h_num_graphs,
		h_nodes,
		h_sources,
		h_targets,
		h_node_batch,
		h_edge_batch
	);
	embed();
	for(uint32_t i = 0; i < LAYERS; i++) {
		residual(i);
	}
	global_mean_pool();
	mm(
		&hidden_heuristic,
		pooled_desc,
		pooled,
		HIDDEN_WEIGHT_DESC,
		HIDDEN_WEIGHTS,
		hidden_desc,
		hidden
	);
	hidden_bias_relu();
	mm(
		&output_heuristic,
		hidden_desc,
		hidden,
		OUTPUT_WEIGHT_DESC,
		OUTPUT_WEIGHTS,
		output_desc,
		d_download
	);
	download(h_results);
}

#ifdef TEST
#include <iostream>
#include <thread>
#include <vector>
#include "example.h"

static void go() {
	float output[EXAMPLE_NUM_GRAPHS];
	model(
		EXAMPLE_NUM_NODES,
		EXAMPLE_NUM_EDGES,
		EXAMPLE_NUM_GRAPHS,
		EXAMPLE_NODES,
		EXAMPLE_SOURCES,
		EXAMPLE_TARGETS,
		EXAMPLE_NODE_BATCH,
		EXAMPLE_EDGE_BATCH,
		output
	);
	for(uint32_t i = 0; i < num_graphs; i++) {
		std::cout << output[i] << " ";
	}
	//std::cout << 'x';
	std::cout << std::endl;
}

int main() {
	init();

	std::vector<std::thread> workers;
	auto f = []() {
		for(int i = 0; i < 100; i++) {
			go();
		}
	};
	for(int i = 0; i < 16; i++) {
		workers.emplace_back(f);
	}
	for(auto &worker : workers) {
		worker.join();
	}
}
#endif
