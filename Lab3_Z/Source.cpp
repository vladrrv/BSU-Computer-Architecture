#include <mpi.h>
#include <iostream>
#include <fstream>
#include <ctime>

using namespace std;

#define N 5
double b[N], x[N], x_ans[N];
double * A = new double[N*N];

void print_matrix() {
	cout << endl;
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j) {
			cout << A[i*N + j] << " ";
		}
		cout << endl;
	}
	cout << endl;
}

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);

	int i, j, k;
	double c, sum = 0.0;
	int rank, nprocs;
	clock_t begin1, end1, begin2, end2;
	MPI_Status status;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);   /* get current process id */
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs); /* get number of processes */

	if (rank == 0)
	{
		for (i = 0; i < N; i++) {
			for (j = 0; j < N; j++) {
				if (i == j)
					A[i*N + j] = 15 + rand() % 10;
				else
					A[i*N + j] = 1 + rand() % 10;
				
			}
		}
		print_matrix();
		ifstream input_file;
		input_file = ifstream("input.txt");
		for (int i = 0; i < N; ++i)
		{
			input_file >> x[i];
			//cout << x[i] << " ";
		}
		for (int i = 0; i < N; ++i)
		{
			b[i] = 0;
			for (int j = 0; j < N; ++j)
			{
				b[i] += A[i*N + j] * x[j];
			}
			//cout << b[i] << " ";
		}
	}

	begin1 = clock();

	MPI_Bcast(&A[0], N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(b, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	for (k = 0; k<N; k++)
	{
		MPI_Bcast(&A[k*N + k], N - k, MPI_DOUBLE, k % nprocs, MPI_COMM_WORLD);
		MPI_Bcast(&b[k], 1, MPI_DOUBLE, k % nprocs, MPI_COMM_WORLD);
		for (i = k + 1; i < N; i++)
		{
			if (i % nprocs == rank)
			{
				c = A[i*N + k] / A[k*N + k];
				for (j = 0; j < N; j++)
				{
					A[i*N + j] = A[i*N + j] - (c * A[k*N + j]);
				}
				b[i] = b[i] - (c * b[k]);
			}
		}
	}

	if (rank == 0)
	{
		cout << "LU:\n";
		print_matrix();
		for (i = N - 1; i >= 0; i--)
		{
			sum = 0;
			for (j = i + 1; j < N; j++)
			{
				sum = sum + A[i*N + j] * x_ans[j];
			}
			x_ans[i] = (b[i] - sum) / A[i*N + i];
		}
		end1 = clock();
	}
	if (rank == 0)
	{
		for (int i = 0; i < N; i++)
		{
			cout << "x_ans = " << x_ans[i] << " x_true = " << x[i] << endl;
		}
		cout << "Time: " << (double)(end1 - begin1) / CLOCKS_PER_SEC << endl;
		system("pause");
	}
	MPI_Finalize();
	return(0);


}