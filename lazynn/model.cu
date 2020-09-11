#include <cassert>
#include <cuda.h>
#include <cublasLt.h>
#include "weights.h"

static const size_t
	ALIGNTO = 256,
	BLOCKS = 32,
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
	*OUT_BIAS = NULL,
	*BACK_BIAS = NULL,
	*HIDDEN_WEIGHTS = NULL,
	*HIDDEN_BIAS = NULL,
	*OUTPUT_WEIGHTS = NULL;

// thread-local global variables, changed once-per-cycle
static thread_local bool thread_initialised = false;
static thread_local uint8_t *p_upload, *d_upload;
static thread_local float *p_download, *d_download;
static thread_local int32_t *nodes = NULL, *batch = NULL;
static thread_local int2 *edges = NULL;
static thread_local float
	*forward_node_norm = NULL,
	*backward_node_norm = NULL,
	*graph_norm = NULL,
	*x = NULL,
	*out = NULL,
	*back = NULL,
	*out_scratch = NULL,
	*back_scratch = NULL,
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
	out_scratch_capacity = 0,
	back_scratch_capacity = 0,
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
	auto offset = blockIdx.x;
	auto channel = threadIdx.x;

	#pragma unroll UNROLL
	for(int i = offset; i < num_nodes; i += BLOCKS) {
		auto index = CHANNELS * i + channel;
		auto node = __ldg(nodes + i);
		auto weight = __ldg(weights + CHANNELS * node + channel);
		__stcg(x + index, weight);
	}
}

static void embed() {
	k_embed<<<BLOCKS, CHANNELS>>>(
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
	float *out,
	float *back
) {
	auto offset = blockIdx.x;
	auto channel = threadIdx.x;

	#pragma unroll UNROLL
	for(int i = offset; i < num_edges; i += BLOCKS) {
		auto edge = __ldg(edges + i);
		auto from = CHANNELS * edge.x + channel;
		auto to = CHANNELS * edge.y + channel;
		atomicAdd(out + to, __ldg(x + from));
		atomicAdd(back + from, __ldg(x + to));
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
	k_gather_neighbours<<<BLOCKS, CHANNELS>>>(
		num_edges,
		x,
		edges,
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
	auto offset = blockIdx.x;
	auto channel = threadIdx.x;

	#pragma unroll UNROLL
	for(int i = offset; i < num_nodes; i += BLOCKS) {
		auto index = CHANNELS * i + channel;
		float forward_norm = __ldg(forward_node_norm + i);
		float backward_norm = __ldg(backward_node_norm + i);
		float out_val = forward_norm * __ldg(out + index);
		float back_val = backward_norm * __ldg(back + index);
		__stcg(out + index, out_val);
		__stcg(back + index, back_val);
	}
}

static void normalise() {
	k_normalise<<<BLOCKS, CHANNELS>>>(
		num_nodes,
		forward_node_norm,
		backward_node_norm,
		out,
		back
	);
}

__global__ void k_bias_relu_sum(
	int32_t num_nodes,
	float *out_bias,
	float *back_bias,
	float *out_scratch,
	float *back_scratch,
	float *x
) {
	auto offset = blockIdx.x;
	auto channel = threadIdx.x;
	float outb = __ldg(out_bias + channel);
	float backb = __ldg(back_bias + channel);

	#pragma unroll UNROLL
	for(int i = offset; i < num_nodes; i += BLOCKS) {
		auto index = CHANNELS * i + channel;
		float current = __ldg(x + index);
		float combined = current +
			fmaxf(0.0f, outb + __ldg(out_scratch + index)) +
			fmaxf(0.0f, backb + __ldg(back_scratch + index));
		__stcg(x + index, combined);
	}
}

static void bias_relu_sum(float *out_bias, float *back_bias) {
	k_bias_relu_sum<<<BLOCKS, CHANNELS>>>(
		num_nodes,
		out_bias,
		back_bias,
		out_scratch,
		back_scratch,
		x
	);
}

__global__ void k_global_mean_pool(
	int32_t num_nodes,
	int32_t num_graphs,
	int32_t *batch,
	float *x,
	float *graph_norm,
	float *pooled
) {
	auto offset = blockIdx.x;
	auto channel = threadIdx.x;

	#pragma unroll UNROLL
	for(int i = offset; i < num_nodes; i += BLOCKS) {
		auto graph = __ldg(batch + i);
		auto norm = __ldg(graph_norm + graph);
		auto value = __ldg(x + CHANNELS * i + channel);
		atomicAdd(pooled + CHANNELS * graph + channel, norm * value);
	}
}

static void global_mean_pool() {
	k_global_mean_pool<<<BLOCKS, CHANNELS>>>(
		num_nodes,
		num_graphs,
		batch,
		x,
		graph_norm,
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
			__ldg(hidden + index) + __ldg(bias + channel)
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
	float *out_bias = OUT_BIAS + CHANNELS * layer;
	float *back_bias = BACK_BIAS + CHANNELS * layer;

	gather_neighbours();
	normalise();
	mm(
		&conv_heuristic, 
		node_desc,
		out,
		CONV_WEIGHT_DESC,
		out_weights,
		node_desc,
		out_scratch
	);
	mm(
		&conv_heuristic, 
		node_desc,
		back,
		CONV_WEIGHT_DESC,
		back_weights,
		node_desc,
		back_scratch
	);
	bias_relu_sum(out_bias, back_bias);
}

static void upload(
	uint32_t h_num_nodes,
	uint32_t h_num_edges,
	uint32_t h_num_graphs,
	const uint32_t *h_nodes,
	const uint32_t *h_sources,
	const uint32_t *h_targets,
	const uint32_t *h_batch
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
	init_matrix_layout(hidden_desc, num_graphs, HIDDEN);
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
	auto graph_norm_offset = ALIGN(
		backward_node_norm_offset + num_nodes * sizeof(float)
	);
	auto batch_offset = ALIGN(
		graph_norm_offset + num_graphs * sizeof(float)
	);
	auto upload_bytes = ALIGN(
		batch_offset + num_nodes * sizeof(int32_t)
	);

	DEVICE_ALLOC(d_upload, upload_bytes);
	DEVICE_ALLOC(d_download, num_graphs);
	DEVICE_ALLOC(x, num_nodes * CHANNELS);
	DEVICE_ALLOC(out, num_nodes * CHANNELS);
	DEVICE_ALLOC(back, num_nodes * CHANNELS);
	DEVICE_ALLOC(out_scratch, num_nodes * CHANNELS);
	DEVICE_ALLOC(back_scratch, num_nodes * CHANNELS);
	DEVICE_ALLOC(pooled, num_graphs * CHANNELS);
	DEVICE_ALLOC(hidden, num_graphs * HIDDEN);
	PAGE_ALLOC(p_upload, upload_bytes);
	PAGE_ALLOC(p_download, num_graphs);
	CUDAOK(cudaMemsetAsync(
		pooled,
		0,
		num_graphs * CHANNELS * sizeof(float),
		CU_STREAM_PER_THREAD
	));

	// alignment-safe: should be aligned from before
	auto p_nodes = (int32_t *)(p_upload + node_offset);
	auto p_edges = (int2 *)(p_upload + edge_offset);
	auto p_forward_node_norm =
		(float *)(p_upload + forward_node_norm_offset);
	auto p_backward_node_norm =
		(float *)(p_upload + backward_node_norm_offset);
	auto p_graph_norm = (float *)(p_upload + graph_norm_offset);
	auto p_batch = (int32_t *)(p_upload + batch_offset);

	memset(p_forward_node_norm, 0, num_nodes * sizeof(float));
	memset(p_backward_node_norm, 0, num_nodes * sizeof(float));
	memset(p_graph_norm, 0, num_graphs * sizeof(float));
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
		p_batch[i] = h_batch[i];
		p_graph_norm[p_batch[i]] += 1.0;
		p_forward_node_norm[i] = 1.0f /
			(1.0f + p_forward_node_norm[i]);
		p_backward_node_norm[i] = 1.0f /
			(1.0f + p_backward_node_norm[i]);
	}
	for(uint32_t i = 0; i < num_graphs; i++) {
		p_graph_norm[i] = 1.0f / p_graph_norm[i];
	}

	CUDAOK(cudaMemcpyAsync(
		d_upload,
		p_upload,
		upload_bytes,
		cudaMemcpyHostToDevice,
		CU_STREAM_PER_THREAD
	));

	nodes = (int32_t *)(d_upload + node_offset);
	edges = (int2 *)(d_upload + edge_offset);
	forward_node_norm = (float *)(d_upload + forward_node_norm_offset);
	backward_node_norm = (float *)(d_upload + backward_node_norm_offset);
	graph_norm = (float *)(d_upload + graph_norm_offset);
	batch = (int32_t *)(d_upload + batch_offset);
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
		h_results[i] = p_download[i];
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
		&OUT_BIAS,
		OUT_BIAS_DATA,
		sizeof(OUT_BIAS_DATA)
	);
	upload_weights(
		&BACK_BIAS,
		BACK_BIAS_DATA,
		sizeof(BACK_BIAS_DATA)
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

#include <cstdio>
extern "C" void model(
	uint32_t h_num_nodes,
	uint32_t h_num_edges,
	uint32_t h_num_graphs,
	const uint32_t *h_nodes,
	const uint32_t *h_sources,
	const uint32_t *h_targets,
	const uint32_t *h_batch,
	float *h_results
) {
	upload(
		h_num_nodes,
		h_num_edges,
		h_num_graphs,
		h_nodes,
		h_sources,
		h_targets,
		h_batch
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
