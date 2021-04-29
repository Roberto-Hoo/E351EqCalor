/* 3.5 Aplicação: Equação do calor
 Nesta seção, vamos construir um código MPI para a resolução da equação do calor
 pelo método das diferenças finitas. Como um caso teste, vamos considerar a equação do calor

		(3.15) U_t(x,t) = c * U_xx(x,t) ,       0 < x < L,  t > 0,  c > 0 (neste exemplo L=1, c=1)
com condições de contorno
		(3.16)    U(0,t) = U(L,t) = 0, t>0
e condições iniciais
		(3.17)    U(x,0) = sen(x*pi),   0 < x < 1

       U(i,m+1)= U(x(i),t(m+1)),
       x(i) = i * hx, hx = L/I,  i=0,1,2,..,I , grid espacial
       t(m) = m*ht, ht = 0.001,  m = 0,1,.., M
       Aproximação por diferenças finitas (Forward-Difference method)
       é estável se 0 < c*(ht/hx^2) = cfl < 1/2

       U(i,m+1) = U(i,m) + (ht/hx^2)*[ U(i-1,m) - 2U(i,m) + U(i+1,m)]
.*/

/*
  size_t  TIPO
  O tipo size_t armazena o tamanho de qualquer tipo de objeto, em bytes.
  É um apelido para um tipo inteiro que aceita apenas valores positivos (unsigned).
  É o tipo retornado pelo operador sizeof.
 */
#include <stdio.h>
#include <math.h>
#include <mpi.h> // API MPI

// tamanho dos passos discretos
double ht = 0.0005;  
double hx = 0.1;
size_t L = 1; // 0< x < L = x_maximo


int world_size; // numero total de processos
int world_rank;// ID (rank) do processo
// parâmetros,  size_t eh int >= 0
size_t M;  // tm = t(m) = m * ht,  m = 0,1,..,M , quantidade de passos no tempo
size_t I;   // xi = x(i) = i * hx,  i = 0,1,..,I, tamanho da malha em x, hx=L/I
bool debug = true;

double funcao(double x, double t) {
    return sin(M_PI * x) * exp(-t * M_PI * M_PI);
}

int main(int argc, char **argv) {

    MPI_Init(NULL, NULL);  // Inicializa o MPI
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);


    I = (size_t)(L / hx);
    double cfl = ht / (hx * hx);

    if ((debug) && (world_rank==0)) {
        printf("CFl = %18.14f  e  I = %18.14f",cfl,(double)I);
    }

    // malha espacial local
    size_t ip = world_rank * int(I / world_size);
    size_t fp = (world_rank + 1) * int(I / world_size);
    if (world_rank == world_size - 1)
        fp = I;
    size_t my_I = fp - ip;

    double x[my_I + 1];
    for (size_t j = 0; j <= my_I; j++)
        x[j] = (ip + j) * hx;

    // solução local
    double u0[my_I + 1], u[my_I + 1];

    // condição inicial
    for (size_t j = 0; j <= my_I; j++) {
        u0[j] = sin(M_PI * x[j]);
    }
    // condições de contorno de Dirichlet
    if (world_rank == 0)
        u[0] = 0.0;
    if (world_rank == world_size - 1)
        u[my_I] = 0.0;

    // auxiliares
    double u00 = 0.0;
    double u0I;

    // iterações no tempo
    for (size_t m = 0; m < M; m++) {

        if (world_rank == 0) {
            MPI_Send(&u0[my_I], 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&u0I, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else if (world_rank < world_size - 1) {
            MPI_Recv(&u00, 1, MPI_DOUBLE, world_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&u0[my_I], 1, MPI_DOUBLE, world_rank + 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&u0I, 1, MPI_DOUBLE, world_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&u0[1], 1, MPI_DOUBLE, world_rank - 1, 0, MPI_COMM_WORLD);
        } else {
            MPI_Recv(&u00, 1, MPI_DOUBLE, world_size - 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&u0[1], 1, MPI_DOUBLE, world_size - 2, 0, MPI_COMM_WORLD);
        }
        // atualização
        u[1] = u0[1] + cfl * (u00 - 2 * u0[1] + u0[2]);
        for (size_t j = 2; j < my_I; j++)
            u[j] = u0[j] + cfl * (u0[j - 1] - 2 * u0[j] + u0[j + 1]);
        if (world_rank < world_size - 1)
            u[my_I] = u0[my_I] + cfl * (u0[my_I - 1] - 2 * u0[my_I] + u0I);

        // prepara nova iteraçã0
        for (size_t j = 0; j <= my_I; j++)
            u0[j] = u[j];

        if ((world_rank == world_size / 2 - 1) && ((m % 100000) == 0)) {
            printf("\nIteracao %d, \nsolucao aproximada %15.12f,  solucao exata  %15.12f,  "\
                          "erro relativo percentual= %15.12f", \
                 m, u[my_I], funcao(0.5, m * ht), \
                 100 * (funcao(0.5, m * ht) - u[my_I]) / funcao(0.5, m * ht));
        }
    }


    // Finaliza o MPI
    MPI_Finalize();

    return 0;
}