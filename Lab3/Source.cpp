#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <ctime>
#include <chrono>
#include <conio.h>
using namespace std;

#define PATH_TAG 103

struct timer {
	clock_t begin;
	void start()
	{
		cout << "Timer started!" << endl;
		begin = clock();
	}
	void print_elapsed()
	{
		clock_t end = clock();
		double elapsed_ms = double(end - begin) * 1000.0 / CLOCKS_PER_SEC;
		cout << "Elapsed time [ctime]: " << elapsed_ms << " ms\n";
	}
};

void generate_matrix(char * name, int n, bool verbose = true)
{
	if (verbose) {
		cout << "Generating matrix... ";
	}
	ofstream output_file(name);
	output_file << n << endl;
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			double val = (((double)(rand() % 2000) / 100) - 10);
			if (i == j) {
				val += 20;
			}
			output_file << val << " ";
		}
		output_file << endl;
	}
	if (verbose) {
		cout << "Done!\n";
	}
}

void ask_generate()
{
	cout << "Generate input file? y/[n]\n";
	if (_getch() == 'y')
	{
		int dim;
		char name[256];
		cout << "Enter file name: ";
		cin >> name;
		cout << "Enter matrix dim: ";
		cin >> dim;
		generate_matrix(name, dim);
	}
}

void print_matrix(char * title, double * matrix, int n) {
	cout << "\n " << title << ":\n";
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j) {
			char num[32];
			sprintf_s(num, "%.4f", matrix[i*n + j]);
			cout << setw(8) << num << " ";
		}
		cout << endl;
	}
	cout << endl;
}

void ask_print(double * matrix, int n, char * title) {
	cout << "Print " << title << "? y/[n]" << endl;
	_getch();
	if (_getch() == 'y') {
		print_matrix(title, matrix, n);
	}
}

double * read(char * path, int & n, bool verbose = false)
{
	ifstream input_file(path);
	if (!input_file.is_open()) 
		cout << "Error opening file: " << path << endl;
	if (verbose) {
		cout << "Reading matrix... ";
	}
	input_file >> n;
	double * matrix = new double[n*n];
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			input_file >> matrix[i*n + j];
		}
	}
	input_file.close();
	if (verbose) {
		cout << " Done!\n";
		cout << "Matrix size: " << n << " x " << n << endl;
	}
	return matrix;
}

void request_file(char * path)
{
	ifstream input_file;
	while (true)
	{
		cout << "Specify input file path: ";
		cin >> path;
		input_file = ifstream(path);
		if (input_file.is_open()) break;
		else cout << "Error opening file!\n";
	}
	input_file.close();
}

void forw_elim(double *origin, double *master_row, int n)
{
	if (*origin == 0)
		return;
	double k = *origin / master_row[0];
	for (int i = 1; i < n; i++) {
		origin[i] -= k * master_row[i];
	}
	*origin = k;
}

void decomp(double * a, int n, int myrank, int ranksize) 
{
	for (int i = 0; i < n - 1; ++i) {
		double *diag_row = &a[i * n + i];
		for (int j = i + 1; j < n; j++) {
			if (j % ranksize == myrank) {
				double * save = &a[j * n + i];
				forw_elim(save, diag_row, n - i);
			}
		}
		for (int j = i + 1; j < n; j++) {
			double * save = &a[j * n + i];
			MPI_Bcast(save, n - i, MPI_DOUBLE, j % ranksize, MPI_COMM_WORLD);
		}
	}
}

void check(int n, double * lu, double * src)
{
	cout << "Checking... ";
	double max_err = 0;
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			double aij = 0, sij = src[i*n + j];
			for (int k = 0; k < n; ++k) {
				double lik = 0, ukj = 0;
				if (i == k) {
					lik = 1;
				}
				if (i > k) {
					lik = lu[i*n + k];
				}
				if (k <= j) {
					ukj = lu[k*n + j];
				}
				aij += lik * ukj;
			}
			//cout << aij << " ";
			double err = abs(aij - sij);
			if (err > max_err) {
				max_err = err;
			}
		}
		//cout << endl;
	}
	char buf[256];
	sprintf_s(buf, "%.5f", max_err);
	cout << "Max error: " << buf << endl;
}

void main(int argc, char ** argv)
{
	MPI_Init(&argc, &argv);

	int myrank, ranksize;
	MPI_Status status;

	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &ranksize);

	timer t;
	char path[256];

	if (myrank == 0) {
		ask_generate();
		request_file(path);
		for (int k = 1; k < ranksize; ++k) {
			MPI_Send(path, 256, MPI_CHAR, k, PATH_TAG, MPI_COMM_WORLD);
		}
	}
	else {
		MPI_Recv(path, 256, MPI_CHAR, 0, PATH_TAG, MPI_COMM_WORLD, &status);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	int n;
	int * indx;
	double * matrix, *matrix_src;
	matrix = read(path, n, myrank == 0);
	indx = new int[n];

	if (myrank == 0) {
		matrix_src = new double[n*n];
		size_t sz = sizeof(double)*n*n;
		memcpy_s(matrix_src, sz, matrix, sz);
		ask_print(matrix_src, n, "source");
		t.start();
	}

	decomp(matrix, n, myrank, ranksize);

	MPI_Barrier(MPI_COMM_WORLD); 

	if (myrank == 1) {
		cout << "Computed!" << endl;
	}
	MPI_Barrier(MPI_COMM_WORLD);

	if (myrank == 0) {
		t.print_elapsed();
		check(n, matrix, matrix_src);
		ask_print(matrix, n, "result");
		system("pause");
	}

	MPI_Finalize();
}


