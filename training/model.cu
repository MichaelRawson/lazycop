#include <iostream>
#include <thread>
#include <vector>

#include "weights.h"
static float *embedding_weights;
static float *out_weights;
static float *back_weights;

void init() {
	cudaMalloc(&embedding_weights, sizeof(EMBEDDING_WEIGHTS));
	cudaMalloc(&out_weights, sizeof(OUT_WEIGHTS));
	cudaMalloc(&back_weights, sizeof(BACK_WEIGHTS));
	cudaMemcpy(
		embedding_weights,
		EMBEDDING_WEIGHTS,
		NODE_TYPES * CHANNELS * sizeof(float),
		cudaMemcpyHostToDevice
	);
	cudaMemcpy(
		out_weights,
		OUT_WEIGHTS,
		CHANNELS * CHANNELS * sizeof(float),
		cudaMemcpyHostToDevice
	);
	cudaMemcpy(
		back_weights,
		BACK_WEIGHTS,
		CHANNELS * CHANNELS * sizeof(float),
		cudaMemcpyHostToDevice
	);
}

__global__ void k_degree(
	int32_t *sources,
	int32_t *targets,
	int32_t *degree,
	int32_t *degree_t
) {
	int32_t edge = blockIdx.x;
	atomicAdd(&degree[targets[edge]], 1);
	atomicAdd(&degree_t[sources[edge]], 1);
}

__global__ void k_embedding(
	float *embedding_weights,
	int32_t *nodes,
	float *x
) {
	int32_t node = blockIdx.x;
	int32_t channel = threadIdx.x;
	x[CHANNELS * node + channel] =
		embedding_weights[CHANNELS * nodes[node] + channel];
}

__global__ void k_conv_stage1(
	float *x,
	int32_t *sources,
	int32_t *targets,
	float *out,
	float *back
) {
	int32_t edge = blockIdx.x;
	int32_t channel = threadIdx.x;
	int32_t source = sources[edge];
	int32_t target = targets[edge];
	int32_t source_index = source * CHANNELS + channel;
	int32_t target_index = target * CHANNELS + channel;
	atomicAdd(&out[target_index], x[source_index]);
	atomicAdd(&back[source_index], x[target_index]);
}

__global__ void k_conv_stage2(
	float *out,
	float *back,
	int32_t *degree,
	int32_t *degree_t
) {
	int32_t node = blockIdx.x;
	int32_t channel = threadIdx.x;
	int32_t index = CHANNELS * node + channel;
	out[index] /= degree[node] + 1;
	back[index] /= degree_t[node] + 1;
}

__global__ void k_conv_stage3(
	float *out_weights,
	float *back_weights,
	float *out,
	float *back,
	float *x
) {
	int32_t node = blockIdx.x;
	int32_t channel = threadIdx.x;
	int32_t index = CHANNELS * node + channel;
	int32_t offset = CHANNELS * channel;

	float total = 0.0;
	#pragma unroll
	for(int i = 0; i < CHANNELS; i++) {
		total += out_weights[offset + i] * out[i];
		total += back_weights[offset + i] * back[i];
	}

	x[index] = total;
}

void embed(
	int32_t num_nodes,
	int32_t *nodes,
	float *x
) {
	k_embedding<<<num_nodes, CHANNELS>>>(
		embedding_weights,
		nodes,
		x
	);
}

void conv(
	int32_t num_nodes,
	int32_t num_edges,
	float *x,
	int32_t *sources,
	int32_t *targets,
	int32_t *degree,
	int32_t *degree_t,
	float *out_weights,
	float *back_weights,
	float *out,
	float *back
) {
	cudaMemcpy(
		out,
		x,
		num_nodes * CHANNELS * sizeof(float),
		cudaMemcpyDeviceToDevice
	);
	cudaMemcpy(
		back,
		x,
		num_nodes * CHANNELS * sizeof(float),
		cudaMemcpyDeviceToDevice
	);
	k_conv_stage1<<<num_edges, CHANNELS>>>(
		x,
		sources,
		targets,
		out,
		back
	);
	k_conv_stage2<<<num_nodes, CHANNELS>>>(
		out,
		back,
		degree,
		degree_t
	);
	k_conv_stage3<<<num_nodes, CHANNELS>>>(
		out_weights,
		back_weights,
		out,
		back,
		x
	);
}

void upload(
	const int32_t *h_nodes,
	const int32_t *h_sources,
	const int32_t *h_targets,
	const int32_t *h_batch,
	int32_t num_nodes,
	int32_t num_edges,
	int32_t num_graphs,
	int32_t **nodes,
	int32_t **sources,
	int32_t **targets,
	int32_t **batch,
	int32_t **degree,
	int32_t **degree_t,
	float **x,
	float **out,
	float **back
) {
	cudaMalloc(nodes, num_nodes * sizeof(int32_t));
	cudaMalloc(sources, num_edges * sizeof(int32_t));
	cudaMalloc(targets, num_edges * sizeof(int32_t));
	cudaMalloc(batch, num_nodes * sizeof(int32_t));
	cudaMalloc(degree, num_nodes * sizeof(int32_t));
	cudaMalloc(degree_t, num_nodes * sizeof(int32_t));
	cudaMalloc(x, num_nodes * CHANNELS * sizeof(float));
	cudaMalloc(out, num_nodes * CHANNELS * sizeof(float));
	cudaMalloc(back, num_nodes * CHANNELS * sizeof(float));
	cudaMemcpy(
		*nodes,
		h_nodes,
		num_nodes * sizeof(int32_t),
		cudaMemcpyHostToDevice
	);
	cudaMemcpy(
		*sources,
		h_sources,
		num_edges * sizeof(int32_t),
		cudaMemcpyHostToDevice
	);
	cudaMemcpy(
		*targets,
		h_targets,
		num_edges * sizeof(int32_t),
		cudaMemcpyHostToDevice
	);
	cudaMemcpy(
		*batch,
		h_batch,
		num_nodes * sizeof(int32_t),
		cudaMemcpyHostToDevice
	);

	cudaMemset(*degree, 0, num_nodes * sizeof(int32_t));
	cudaMemset(*degree_t, 0, num_nodes * sizeof(int32_t));
	k_degree<<<num_edges, 1>>>(
		*sources,
		*targets,
		*degree,
		*degree_t
	);
}

float download(
	int32_t *nodes,
	int32_t *sources,
	int32_t *targets,
	int32_t *batch,
	int32_t *degree,
	int32_t *degree_t,
	float *x,
	float *out,
	float *back
) {
	float result;
	cudaMemcpy(
		&result,
		x,
		sizeof(float),
		cudaMemcpyDeviceToHost
	);

	cudaFree(nodes);
	cudaFree(sources);
	cudaFree(targets);
	cudaFree(batch);
	cudaFree(degree);
	cudaFree(degree_t);
	cudaFree(x);
	cudaFree(out);
	cudaFree(back);
	return result;
}

extern "C" float model(
	const int32_t *h_nodes,
	const int32_t *h_sources,
	const int32_t *h_targets,
	const int32_t *h_batch,
	int32_t num_nodes,
	int32_t num_edges,
	int32_t num_graphs
) {
	int32_t *nodes, *sources, *targets, *batch, *degree, *degree_t;
	float *x, *out, *back;
	upload(
		h_nodes,
		h_sources,
		h_targets,
		h_batch,
		num_nodes,
		num_edges,
		num_graphs,
		&nodes,
		&sources,
		&targets,
		&batch,
		&degree,
		&degree_t,
		&x,
		&out,
		&back
	);
	embed(num_nodes, nodes, x);
	conv(
		num_nodes,
		num_edges,
		x,
		sources,
		targets,
		degree,
		degree_t,
		out_weights,
		back_weights,
		out,
		back
	);
	for(int i = 0; i < 4; i++) {
		int32_t offset = CHANNELS * CHANNELS * (2 * i + 1);
		conv(
			num_nodes,
			num_edges,
			x,
			sources,
			targets,
			degree,
			degree_t,
			out_weights + offset,
			back_weights + offset,
			out,
			back
		);
	}
	return download(
		nodes,
		sources,
		targets,
		batch,
		degree,
		degree_t,
		x,
		out,
		back
	);
}

int main() {
	init();

	const int num_nodes = 1000;
	const int num_edges = 2000;
	const int num_graphs = 10;
	int32_t nodes[num_nodes];
	int32_t sources[num_edges];
	int32_t targets[num_edges];
	int32_t batch[num_nodes];
	for(int i = 0; i < num_nodes; i++) {
		nodes[i] = i % 7;
		batch[i] = i / (num_nodes / num_graphs);
	}
	for(int i = 0; i < num_edges; i++) {
		sources[i] = i / 2;
		targets[i] = i / 2;
	}

	auto f = [nodes, sources, targets, batch]() {
		for(int i = 0; i < 10; i++) {
			float result = model(
				nodes,
				sources,
				targets,
				batch,
				num_nodes,
				num_edges,
				num_graphs
			);
			std::cout << result << std::endl;
		}
	};
	std::vector<std::thread> workers;
	for(int i = 0; i < 16; i++) {
		workers.emplace_back(f);
	}
	for(auto &worker : workers) {
		worker.join();
	}
}
