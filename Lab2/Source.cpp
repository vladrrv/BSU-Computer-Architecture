#include <mpi.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <chrono>
#include <conio.h>
using namespace std;

void generate_matrix_vector(char * name, int m, int n) {
	cout << "Generating...\n";
	ofstream output_file(name);
	output_file << m << " " << n << endl;
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			output_file << ( ( (double)(rand() % 2000) / 100 ) - 10 ) << " ";
		}
		output_file << endl;
	}
	cout << "Done generating matrix!\n";
	for (int j = 0; j < n; ++j)
	{
		output_file << (((double)(rand() % 2000) / 100) - 10) << endl;
	}
	output_file.close();
	cout << "Done generating vector!\n";
}

double * compute_partial_sum(int myrank, int ntasks, int m, int n, double * matrix, double * vector)
{
	double * partial_sum_vector = new double[m];
	fill_n(partial_sum_vector, m, 0);
	for (int i = 0; i < m; ++i)
	{
		for (int j = myrank; j < n; j += ntasks)
		{
			partial_sum_vector[i] += matrix[i*n+j] * vector[j];
		}
	}
	return partial_sum_vector;
}

void print_matrix(char * title, double * matrix, int m, int n) {
	cout << "\n " << title << ":\n";
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < n; ++j) {
			cout << matrix[i*n+j] << " ";
		}
		cout << endl;
	}
}

#define RESULT_TAG 100
#define M_TAG 101
#define N_TAG 102
#define MATRIX_TAG 103
#define VECTOR_TAG 104

void main(int argc, char ** argv)
{
	MPI_Init(&argc, &argv);       /* initialize MPI system */

	int myrank, ranksize;
	MPI_Status status;

	int m, n;
	double * matrix;
	double * vector;

	auto start = chrono::high_resolution_clock::now();
	clock_t begin;


	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);    /* my place in MPI system */
	MPI_Comm_size(MPI_COMM_WORLD, &ranksize);  /* size of MPI system */

	if (myrank == 0)               /* I am the master */
	{
		cout << "Generate input file? y/[n]\n";
		if (_getch() == 'y')
		{
			int dim1, dim2;
			char name[256];
			cout << "Enter file name: ";
			cin >> name;
			cout << "Enter matrix dims: ";
			cin >> dim1 >> dim2;
			generate_matrix_vector(name, dim1, dim2);
		}
		char path[256];
		ifstream input_file;
		while (true)
		{
			cout << "Specify input file path: ";
			cin >> path;
			input_file = ifstream(path);
			if (input_file.is_open()) break;
			else cout << "Error opening file!\n";
		}

		input_file >> m >> n;
		matrix = new double[m*n];
		vector = new double[n];
		cout << "Reading file... ";
		// Reading matrix
		for (int i = 0; i < m; ++i)
		{
			for (int j = 0; j < n; ++j) {
				input_file >> matrix[i*n+j];
			}
		}
		// Reading vector
		for (int j = 0; j < n; ++j) {
			input_file >> vector[j];
		}
		cout << "Finished!\n";

		//print_matrix("Matrix", matrix, m, n);
		//print_matrix("Vector", vector, n, 1);

	}

	MPI_Barrier(MPI_COMM_WORLD);  /* make sure all MPI tasks are running */

	if (myrank == 0)               /* I am the master */
	{

		begin = clock();
		start = chrono::high_resolution_clock::now();
		/* distribute parameter */
		cout << "Master: Sending n, m, matrix and vector to MPI-Processes \n";
		for (int k = 1; k < ranksize; k++)
		{
			MPI_Send(&m, 1, MPI_LONG, k, M_TAG, MPI_COMM_WORLD);
			MPI_Send(&n, 1, MPI_LONG, k, N_TAG, MPI_COMM_WORLD);
			MPI_Send(matrix, m*n, MPI_DOUBLE, k, MATRIX_TAG, MPI_COMM_WORLD);
			MPI_Send(vector, n, MPI_DOUBLE, k, VECTOR_TAG, MPI_COMM_WORLD);
		}
	}
	else {	/* I am a slave */
			/* receive parameters */
		MPI_Recv(&m, 1, MPI_LONG, 0, M_TAG, MPI_COMM_WORLD, &status);
		MPI_Recv(&n, 1, MPI_LONG, 0, N_TAG, MPI_COMM_WORLD, &status);
		matrix = new double[m*n];
		vector = new double[n];
		MPI_Recv(matrix, m*n, MPI_DOUBLE, 0, MATRIX_TAG, MPI_COMM_WORLD, &status);
		MPI_Recv(vector, n, MPI_DOUBLE, 0, VECTOR_TAG, MPI_COMM_WORLD, &status);
	}

	/* compute my portion */
	double * result = compute_partial_sum(myrank, ranksize, m, n, matrix, vector);
	MPI_Barrier(MPI_COMM_WORLD);

	if (myrank == 0)	/* I am the master */
						/* collect results, add up, and print results */
	{
		for (int k = 1; k < ranksize; k++)
		{
			double * partial_sum_vector = new double[m];
			MPI_Recv(partial_sum_vector, m, MPI_DOUBLE, k, RESULT_TAG, MPI_COMM_WORLD, &status);
			for (int i = 0; i < m; ++i) {
				result[i] += partial_sum_vector[i];
			}
		}
		cout << "Master: Has collected sum from MPI-Processes \n";
		clock_t end = clock();
		double elapsed_ms = double(end - begin) * 1000.0 / CLOCKS_PER_SEC;
		cout << "Elapsed time [ctime]: " << elapsed_ms << " ms\n";

		auto finish = chrono::high_resolution_clock::now();
		chrono::duration<double> elapsed = finish - start;
		cout << "Elapsed time [chrono]: " << elapsed.count() * 1000 << " ms\n";

		MPI_Barrier(MPI_COMM_WORLD);


		cout << "Print result? y/[n]\n";
		_getch();
		if (_getch() == 'y')
		{
			print_matrix("Result", result, m, 1);
		}
		system("pause");
	}
	else {	/* I am a slave */
			/* send my result back to master */
		MPI_Send(result, m, MPI_DOUBLE, 0, RESULT_TAG, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
	}

	MPI_Finalize();
}