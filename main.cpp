#include <iostream>


int main(){
    int m = 4;
    int n = 3;

    // arr[n,m]
    int **arr = (int **)malloc(n * sizeof(int*));

    for (int i=0; i<n; ++i){
        arr[i] = (int*)malloc(m * sizeof(int));
    }

    for (int i=0; i<n; ++i){
        free(arr[i]);
    }

    free(arr);

}