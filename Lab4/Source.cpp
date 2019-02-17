#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <ctime>
#include <chrono>
#include <conio.h>
using namespace std;


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


double u_true(double x, double y) 
{
	//return x*x*x + x*y*y + y*y*y;
	return sin(x+1)*x - cos(y+1)*y;
}

double u_true(int n, int i, int j)
{
	double h = 1.0/n, x = h * i, y = h * j;
	return u_true(x, y);
}

double f(double x, double y)
{
	//return 8*x + 6*y;
	return 2*cos(x+1) - sin(x+1)*x + cos(y+1)*y + 2*sin(y+1);
}

double f(int n, int i, int j)
{
	double h = 1.0/n, x = h * i, y = h * j;
	return f(x, y);
}

double ** gauss_zeid(int n, double eps, int rank, int ranksize)
{
	int rows_per_proc = ceil(double(n-1) / ranksize),
		start_row = 1 + rank * rows_per_proc,
		end_row = min({ (rank+1) * rows_per_proc, n-1 });
	double h = 1.0 / n;
	double ** u = new double*[n + 1];

	for (int i = 0; i < n + 1; ++i) {
		u[i] = new double[n + 1];
		fill_n(u[i], n + 1, 0);
		u[i][0] = u_true(n, i, 0);
		u[i][n] = u_true(n, i, n);
	}
	for (int j = 0; j < n + 1; ++j) {
		u[0][j] = u_true(n, 0, j);
		u[n][j] = u_true(n, n, j);
	}

	double dmax = 0, d = 0;
	do {
		dmax = 0;
		d = 0;
		if (rank % 2 == 0) {
			if (rank > 0) {
				MPI_Send(u[start_row], n+1, MPI_DOUBLE, rank-1, start_row, MPI_COMM_WORLD);
				MPI_Recv(u[start_row-1], n+1, MPI_DOUBLE, rank-1, start_row-1, MPI_COMM_WORLD, NULL);
			}
			if (rank < ranksize - 1) {
				MPI_Send(u[end_row], n+1, MPI_DOUBLE, rank+1, end_row, MPI_COMM_WORLD);
				MPI_Recv(u[end_row+1], n+1, MPI_DOUBLE, rank+1, end_row+1, MPI_COMM_WORLD, NULL);
			}
		}
		else {
			if (rank > 0) {
				MPI_Recv(u[start_row-1], n+1, MPI_DOUBLE, rank-1, start_row-1, MPI_COMM_WORLD, NULL);
				MPI_Send(u[start_row], n+1, MPI_DOUBLE, rank-1, start_row, MPI_COMM_WORLD);
			}
			if (rank < ranksize - 1) {
				MPI_Recv(u[end_row+1], n+1, MPI_DOUBLE, rank+1, end_row+1, MPI_COMM_WORLD, NULL);
				MPI_Send(u[end_row], n+1, MPI_DOUBLE, rank+1, end_row, MPI_COMM_WORLD);
			}
		}
		for (int i = start_row; i <= end_row; ++i) {

			for (int j = 1; j < n; ++j) {
				double temp = u[i][j];
				u[i][j] = 0.25*(u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1] - h*h*f(n, i, j));
				double dm = abs(temp - u[i][j]);
				if (d < dm) d = dm;
			}
		}
		MPI_Allreduce(&d, &dmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	} while (dmax > eps);

	// Processes 1,2,... send results, process 0 gathers them 
	if (rank > 0) {
		for (int i = start_row; i <= end_row; ++i) {
			MPI_Send(u[i], n + 1, MPI_DOUBLE, 0, i, MPI_COMM_WORLD);
		}
	}
	else {
		for (int k = 1; k < ranksize; ++k) {
			int s_r = 1 + k * rows_per_proc,
				e_r = min({ (k + 1) * rows_per_proc, n - 1 });
			for (int i = s_r; i <= e_r; ++i) {
				MPI_Recv(u[i], n+1, MPI_DOUBLE, k, i, MPI_COMM_WORLD, NULL);
			}
		}
	}
	return u;
}

void check(double ** u, int n)
{
	cout << "Checking... ";
	double max_err = 0;
	for (int i = 0; i < n + 1; ++i)
		for (int j = 0; j < n + 1; ++j) {
			double e = abs(u_true(n, i, j) - u[i][j]);
			if (max_err < e) max_err = e;
		}
	cout << "Max error: " << max_err << endl;
}

void save(double ** u, int n)
{
	ofstream output("u.txt");
	cout << "Saving... ";
	double max_err = 0;
	for (int i = 0; i < n + 1; ++i) {
		for (int j = 0; j < n + 1; ++j) {
			output << u[i][j] << " ";
		}
		output << endl;
	}
	output.close();
	cout << "Done!" << endl;
}

void main(int argc, char ** argv)
{
	MPI_Init(&argc, &argv);

	int myrank, ranksize;
	MPI_Status status;

	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &ranksize);

	timer t;
	double ** u = 0, eps = 0.0001;
	int n = 100;

	MPI_Barrier(MPI_COMM_WORLD);

	if (myrank == 0) {
		t.start();
	}

	u = gauss_zeid(n, eps, myrank, ranksize);

	if (myrank == 0) {
		t.print_elapsed();
		check(u, n);
		save(u, n);
		system("pause");
	}

	MPI_Finalize();
}


