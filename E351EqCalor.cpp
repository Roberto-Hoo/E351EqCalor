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
double ht = 0.0005; // espaçamento da malha no tempo t
double hx = 0.1;    // espaçamento da malha em x
double L = 1.0;     // 0 < x < L = x_maximo
double TF = 0.5;    // tempo final

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


    I = (size_t) (L / hx);
    M = (size_t) (TF / ht);
    double cfl = ht / (hx * hx);

    if ((debug) && (world_rank == 0)) {
        printf("CFl = %8.3f  ,  I = %8.3f  , M = %8.3f", cfl, (double) I, (double) M);
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

    // solução u0 e u em cada pedaço da malha
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
    for (size_t m = 1; m <= M; m++) {

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

    if (world_rank == 0) {

        double UF[M + 1];
        for (int i = 0; i <= M; i++) //Inicializa a solucao final no processo 0
            UF[i] = 0.0;

        for (int i = 1; i <= my_I; i++) // Coloca no processo 0 a sua solucao
            UF[i] = u[i];

        // Recebe a solucao dos processo 1,2,.. ,world_size-2
        for (int i = 1; i <= world_size - 2; i++)
            MPI_Recv(&UF[i * my_I+1], my_I, MPI_DOUBLE, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Recv(&UF[(world_size - 1) * my_I+1], I - (world_size - 1) * my_I, MPI_DOUBLE,
                 world_size - 1, world_size - 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("\n x_i       exato(x_i,0.5)      u(x_i,0.5)        erro = exato - u  ");
        for (int i = 0; i <= 10; i++)
            printf("\n %2.1f       %12.8f      %12.8f      %12.3es    ",
                   (double) (i) * 0.1, funcao((double)(i)*0.1,0.5), UF[i],
                   funcao((double)(i)*0.1,0.5)-UF[i]);

    } else {
        // Manda para o processo 0 as solucoes feitas em cada pedaço
        MPI_Send(&u[1], my_I, MPI_DOUBLE, 0, world_rank, MPI_COMM_WORLD);
    }

    /*
       if (world_rank == 0) {

        printf("\n  x_i        u(x_i,0.5)       exato(x_i,0.5)    erro = exato - u  ");
        for (int i = 0; i <= 10; i++) {
            printf("\n %2.1f   %10.6", (double) (i) * 0.1), UF[i]);

        }
     */
    // Finaliza o MPI
    MPI_Finalize();

    return 0;
}