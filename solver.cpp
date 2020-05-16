#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "gurobi_c++.h"
#include "Proto\graph.pb.h"
#include <iostream>
#include <fstream>


#define MAXNODES_LEFT 500
#define MAXNODES_RIGHT 500


#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>

#include "bigint/BigIntegerLibrary.hh"

uint64_t timeSinceEpochMillisec() {
	using namespace std::chrono;
	return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

uint32_t corr_matrix[MAXNODES_LEFT][MAXNODES_LEFT];
GRBVar var_matrix[MAXNODES_LEFT][MAXNODES_LEFT];

uint32_t graph_matrix[MAXNODES_LEFT][MAXNODES_RIGHT]; // g[x] ist liste an knoten, zu denen knoten x verbunden ist
uint32_t size_array[MAXNODES_LEFT]; //size[x] gibt an, wie viele knoten zu x verbunden sind

uint32_t score[MAXNODES_LEFT];

uint32_t links = 15, rechts = 15; // left ist anzahl der knoten auf der linken seite (die die permutiert werden sollen) und right auf der anderen Seite
int prob = 40; // wahrscheinlichkeit, dass eine bestimme kante in den zufallsgraphen kommen, 0 <= prob <= 100

void generate_planar_graph_matrix(uint32_t prob)
{
	time_t t;
	srand((unsigned)time(&t));
	for (uint32_t i = 0; i < links; i++) size_array[i] = 0; //size array initialisiert
	uint32_t actual_left = 0;
	uint32_t actual_right = 0;
	while (true)
	{
		//zuf�llige planare kante
		while (rand() % 100 < prob) actual_left++;
		while (rand() % 100 < prob) actual_right++;

		if (actual_left >= links || actual_right >= rechts) return;

		graph_matrix[actual_left][size_array[actual_left]++] = actual_right;
		if (rand() % 100 < 50)
			actual_left++;
		else
			actual_right++;
	}
}

void generate_graph_matrix()
{
	
	for (uint32_t i = 0; i < links; i++) size_array[i] = 0; //size array initialisiert
	for (uint32_t l = 0; l < links; l++) //i gibt an welche kante wir gerade betrachten auf der linken seite
	for (uint32_t r = 0; r < rechts; r++)
		if (rand() % 100 < prob)	graph_matrix[l][size_array[l]++] = r;		
}

void print_graph_matrix()
{
	for (uint32_t l = 0; l < links; l++)
	{
		for (uint32_t i = 0; i < size_array[l]; i++)
			printf("%u ", graph_matrix[l][i]);
		printf("\n");
	}
}

//es ist a(x,y) = #neue Kreuzungen wenn o(x)<o(y), o ist permutation
void generate_corr_matrix()
{	
	for (uint32_t x = 0; x < links; x++) 
	for (uint32_t y = 0; y < links; y++)
	{
		if (x == y) continue;

		corr_matrix[x][y] = 0;

		for (uint32_t i = 0; i < size_array[x]; i++)// index von x
		for (uint32_t j = 0; j < size_array[y]; j++) // index von y
					if (graph_matrix[x][i] > graph_matrix[y][j]) corr_matrix[x][y]++;
	}


}

uint32_t solve_score_ILP(uint32_t node);

void solve_score_ILP()
{
	generate_corr_matrix();

	for (uint32_t i = 0; i < links; i++)
		score[i] = solve_score_ILP(i);

	for (uint32_t i = 0; i < links; i++)
		printf("Wenn %u ganz links ist kann man minimal %u Kreuzungen erreichen\n", i, score[i]);
}

uint32_t solve_score_ILP(uint32_t node)
{
	//0<=node<left

	GRBEnv env = GRBEnv();
	GRBModel model = GRBModel(env);

	uint32_t x, y, z;
	for (x = 0; x < links; x++)
		for (y = 0; y < links; y++)
		{
			if (x == y || x == node || y == node) continue;

			var_matrix[x][y] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY);
		}

	model.update();

	

	GRBLinExpr obj = 0.0;
	for (x = 0; x < links; x++)
	for (y = 0; y < links; y++)
	{
		if (x == y || y == node) continue;

		if (x == node)
			obj += corr_matrix[x][y];
		else
			obj += corr_matrix[x][y] * var_matrix[x][y];
	}

	model.setObjective(obj, GRB_MINIMIZE);

	for (x = 0; x < links; x++)
		for (y = 0; y < links; y++)
		{
			if (x == y || x == node || y == node) continue;

			GRBTempConstr constr = var_matrix[x][y] == 1 - var_matrix[y][x];

			model.addConstr(constr);
		}

	for (x = 0; x < links; x++)
		for (y = 0; y < links; y++)
			for (z = 0; z < links; z++)
			{
				if (x == y || x == z || y == z || x == node || y == node || z == node) continue;

				GRBTempConstr constr = var_matrix[x][y] + var_matrix[y][z] - var_matrix[x][z] <= 1;

				model.addConstr(constr);
			}

	//model.set(GRB_IntParam_Threads, 1);

	model.optimize();

	return (uint32_t) std::lround(model.get(GRB_DoubleAttr_ObjVal));

//	printf("mit %u ganz links kann man %f kreuzungen minimal erreichen\n\n", node, model.get(GRB_DoubleAttr_ObjVal));
}

//time ist die zeit in millisekunden in denen Trainingsdaten generiert werden sollen
void safe_score_ILP_training_sess(std::string path, uint32_t time)
{
	GOOGLE_PROTOBUF_VERIFY_VERSION;

	uint64_t t = timeSinceEpochMillisec();


	instance::Data data;

	{
		std::fstream input(path, std::ios::in | std::ios::binary);

		if (input)
			data.ParseFromIstream(&input);
	}

	while (timeSinceEpochMillisec() - t < time)
	{
		instance::Data_Graph *graph = data.add_graph();

		{
			uint32_t probs[] = { 5,10,20,30,40,50,60,70,80,90 };
			//prob = probs[rand() % 10];
			prob = (rand() % 25) + 5;
			links = (rand() % 15) + 5;
			links = 5;
			rechts = (rand() % 15) + 5;
			rechts = 5;
		}

		generate_graph_matrix();
		solve_score_ILP();

		graph->set_right(rechts);

		for (uint32_t i = 0; i < links; i++)
		{
			instance::Data_Graph_Node *node = graph->add_nodes();

			for (uint32_t j = 0; j < size_array[i]; j++)
				node->add_connectedto(graph_matrix[i][j]);

			graph->add_score(score[i]);
		}
	}

	std::fstream output(path, std::ios::out | std::ios::trunc | std::ios::binary);

	data.SerializeToOstream(&output);
}

void safe_score_ILP_training_sess(std::string path)
{
	while (true)
		safe_score_ILP_training_sess(path, 1000 * 5 * 60);
}


void safe_score_ILP_instance(std::string path)
{
	instance::Data_Graph graph;
	
	graph.set_right(rechts);

	for (uint32_t i = 0; i < links; i++)
	{
		instance::Data_Graph_Node *node = graph.add_nodes();
		
		for (uint32_t j = 0; j < size_array[i]; j++)
			node->add_connectedto(graph_matrix[i][j]);
		
		graph.add_score(score[i]);
	}
	
	std::fstream output(path, std::ios::out | std::ios::trunc | std::ios::binary);

	graph.SerializeToOstream(&output);
}

void read(std::string path)
{
	instance::Data_Graph graph;

	std::fstream input(path, std::ios::in | std::ios::binary);

	graph.ParseFromIstream(&input);
	
	for (uint32_t i = 0; i < graph.score_size(); i++)
		printf("%u\n", graph.score(i));

	for (uint32_t i = 0; i < graph.nodes_size(); i++)
	{
		for (uint32_t j = 0; j < graph.nodes(i).connectedto_size(); j++)
			printf("%u ", graph.nodes(i).connectedto(j));
		printf("\n");
	}
}

GRBConstr constr_reverse[MAXNODES_LEFT][MAXNODES_LEFT];
GRBConstr constr_transitive[MAXNODES_LEFT][MAXNODES_LEFT][MAXNODES_LEFT];

void solve_score_ILP_efficient()
{
	generate_corr_matrix();

	GRBEnv env = GRBEnv();
	GRBModel model = GRBModel(env);

	uint32_t x, y, z;
	for (x = 0; x < links; x++)
		for (y = 0; y < links; y++)
		{
			if (x == y) continue;

			var_matrix[x][y] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY);
		}


	for (x = 0; x < links; x++)
		for (y = 0; y < links; y++)
		{
			if (x == y) continue;

			constr_reverse[x][y] = model.addConstr(var_matrix[x][y] == 1 - var_matrix[y][x]);
		}

	for (x = 0; x < links; x++)
		for (y = 0; y < links; y++)
			for (z = 0; z < links; z++)
			{
				if (x == y || x == z || y == z) continue;

				constr_transitive[x][y][z] = model.addConstr(var_matrix[x][y] + var_matrix[y][z] - var_matrix[x][z] <= 1);
			}

	for (uint32_t node = 0; node < links; node++)
	{
		/*
		//remove unnecessary vars
		for (x = 0; x < links; x++)
		{
			if (x == node)
				continue;

			model.remove(var_matrix[x][node]);
			model.remove(var_matrix[node][x]);
		}

		//remove unnecessary reverse constraint
		for (x = 0; x < links; x++)
		{
			if (x == node)
				continue;

			model.remove(constr_reverse[x][node]);
			model.remove(constr_reverse[node][x]);
		}
		//remove unnecessary transitive constraint
		for (x = 0; x < links; x++)
			for (y = 0; y < links; y++)
			{
				if (x == y || x == node || y == node)
					continue;

				model.remove(constr_transitive[node][x][y]);
				model.remove(constr_transitive[x][node][y]);
				model.remove(constr_transitive[x][y][node]);
			}
		*/
		
		GRBConstr constr_zero[MAXNODES_LEFT];
		GRBConstr constr_one[MAXNODES_LEFT];

		for (x = 0; x < links; x++)
		{
			if (x == node)
				continue;

			constr_zero[x] = model.addConstr(var_matrix[x][node] == 0);
			constr_one[x] = model.addConstr(var_matrix[node][x] == 1);
		}

		model.update();

		GRBLinExpr obj = 0.0;

		for (x = 0; x < links; x++)
			for (y = 0; y < links; y++)
			{
				if (x == y || y == node) continue;

				if (x == node)
					obj += corr_matrix[x][y];
				else
					obj += corr_matrix[x][y] * var_matrix[x][y];
			}

		model.setObjective(obj, GRB_MINIMIZE);

		model.optimize();

		for (x = 0; x < links; x++)
		{
			if (x == node)
				continue;

			model.remove(constr_zero[x]);
			model.remove(constr_one[x]);
		}

		score[node] = (uint32_t)std::lround(model.get(GRB_DoubleAttr_ObjVal));
	}

	for (uint32_t i = 0; i < links; i++)
		printf("Wenn %u ganz links ist kann man minimal %u Kreuzungen erreichen\n", i, score[i]);
}

void solve_ILP()
{
	generate_corr_matrix();

	GRBEnv env = GRBEnv();
	GRBModel model = GRBModel(env);

	GRBVar *var = (GRBVar*) malloc(links * links * sizeof(GRBVar));
	uint32_t x, y, z;
	for (x = 0; x < links; x++)
	for (y = 0; y < links; y++)
	{
		if (x == y) continue;

		var_matrix[x][y] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY);
	}

	model.update();

	GRBLinExpr obj = 0.0;
	for (x = 0; x < links; x++)
	for (y = 0; y < links; y++)
	{
		if (x == y) continue;

		obj += corr_matrix[x][y] * var_matrix[x][y];
	}

	model.setObjective(obj, GRB_MINIMIZE);
	

	for (x = 0; x < links; x++)
	for (y = 0; y < links; y++)
	{
		if (x == y) continue;

		GRBTempConstr constr = var_matrix[x][y] == 1 - var_matrix[y][x];

		model.addConstr(constr);
	}

	for (x = 0; x < links; x++)
	for (y = 0; y < links; y++)
	for (z = 0; z < links; z++)
	{
		if (x == y || x == z || y == z) continue;

		GRBTempConstr constr = var_matrix[x][y] + var_matrix[y][z] - var_matrix[x][z] <= 1;

		model.addConstr(constr);
	}

	model.optimize();
}



//Berechnet a^b
BigInteger power(BigInteger a, BigInteger b)
{
	if (b == 1)
		return a;
	if (b % 2 == 0)
		return power(a * a, b / 2);
	else
		return a * power(a * a, b / 2);
}

//Berechnet a*(a-1)*...*(a-(b-1))
BigInteger restricted_factorial(BigInteger a, BigInteger b)
{
	if (b == 1)
		return a;
	else
		return a * restricted_factorial(a - 1, b - 1);
}

// berechne a �ber b
BigInteger bin_coeff(BigInteger a, BigInteger b)
{
	return restricted_factorial(a, b) / restricted_factorial(b, b);
}


int block_size = 9;
// berechne zufallszahl z mit 1 <= z <= a
BigInteger big_rand(BigInteger a)
{
	std::default_random_engine generator;
	generator.seed((int)timeSinceEpochMillisec());
	
	std::string upper_bound_string = bigIntegerToString(a);
	std::string random_number = "";

	std::uniform_int_distribution<int> distribution(0, 999999999);

	uint32_t num_blocks = (upper_bound_string.length() + (block_size - 1)) / block_size;

	bool bounded = true;
	for (uint32_t block = 0; block < num_blocks; block++)
	{
		int this_block;
		if (bounded)
		{
			int first_bound_length = upper_bound_string.length() % block_size == 0 ? block_size : upper_bound_string.length() % block_size;
			int bound_length = block == 0 ? first_bound_length : block_size;
			int bound_offset = block == 0 ? 0 : first_bound_length + (block - 1) * block_size;

			
			std::string to_convert = upper_bound_string.substr(bound_offset, bound_length);

			int bound = std::stoi(to_convert);

			std::uniform_int_distribution<int> distribution(0, bound);
			
			this_block = distribution(generator);

			if (this_block != bound)
				bounded = false;
		}
		else
		{
			this_block = distribution(generator);
		}
		random_number += std::to_string(this_block);
	}
	BigInteger return_number = stringToBigInteger(random_number);

	if (return_number > a)
		std::cout << "error beim zufallszahlengenerator";
	return return_number == 0 ? big_rand(a) : return_number;
}


//1 <= rand
//return: n, sodass bin_coeff(n-1, k) < rand <= bin_coeff(n, k)
BigInteger get_upper_bin_coeff(BigInteger k, BigInteger rand)
{
	BigInteger exp_fac = 2;

	if (rand == 1)
		return k;

	//invariante: bin_coeff(lower_bound, k) < rand
	BigInteger lower_bound = k;

	//gesucht: upper_bound, sodass rand <= bin_coeff(upper_bound, k) 
	BigInteger maybe_upper_bound = lower_bound * exp_fac;

	while (bin_coeff(maybe_upper_bound, k) < rand)
	{
		lower_bound = maybe_upper_bound;
		maybe_upper_bound = maybe_upper_bound * exp_fac;
	}

	BigInteger upper_bound = maybe_upper_bound;

	while (upper_bound - lower_bound > 1)
	{
		BigInteger middle = (upper_bound + lower_bound) / 2;

		if (bin_coeff(middle, k) >= rand)
			upper_bound = middle;
		else
			lower_bound = middle;
	}
	
	return upper_bound;
}

void generate_graph(uint32_t left, uint32_t right)
{
	links = left;
	rechts = right;
	for (uint32_t i = 0; i < links; i++) size_array[i] = 0; //size array initialisiert

	BigInteger k = left;
	BigInteger n = power(2, right);

	while (k > 0)
	{
		BigInteger upper_bound = bin_coeff(n + k - 1, k);

		BigInteger rand = big_rand(upper_bound);

		BigInteger a = get_upper_bin_coeff(k, rand);

		n = a - k + 1; //das hier ist die zufallszahl
		k--;

		std::cout << n << std::endl;

		int node_representation = n.toInt() - 1;
		int which_node = (k - 1).toInt();
		
		for (int i = 0; i < right; i++)
		{
			if (node_representation & 1 << i)
				graph_matrix[which_node][size_array[which_node]++] = i;
		}
	}
}

int main(int argc, char *argv[])
{
	GOOGLE_PROTOBUF_VERIFY_VERSION;
	
	
	//safe_score_ILP_training_sess("D:/Informatik/6. Semester/BachelorArbeit/Trainingsdaten/training_to_check_real.graphs", 10000);
	
	generate_graph_matrix();

	solve_score_ILP();
	solve_score_ILP_efficient();

	return 0;
}