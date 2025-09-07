#include <iostream>
#include <vector>
#include <cstring>
#include <random>
#include <fstream>



double dot(double* x1, double* x2, int dim){
    double res = 0;
    for(int i=0; i<dim; ++i){
        res += x1[i] * x2[i];
    }
    return res;
}


class LinReg{
    private:
        
        int dim;
        double* weights;

    public:

        LinReg(int d){
            dim = d;
            weights = (double*) calloc(dim + 1, sizeof(double));// init with zeros
        }

        
        // x[dim]
        double pred_single(double* x) const{
            double res = dot(x, weights, dim);
            res += weights[dim]; // adding w_0
            return res;
        }


        // X[n_obj][dim]
        double* pred_batch(double** X, int n_obj) const{
            double* res = (double*) calloc(n_obj, sizeof(double));
            for(int i=0; i<n_obj; ++i){
                res[i] = pred_single(X[i]);
            }
            return res;
        }


        // return w from R^(dim+1) where w_0 = w[dim] = bias term
        double* GetWeights() const{
            double* w = (double*) calloc(dim + 1, sizeof(double));
            memcpy(w, weights, (dim + 1) * sizeof(double));
            return w;
        }


        // set weights to w where w[dim] = w_0 = bias term
        void SetWeights(double *w){
            memcpy(weights, w, (dim + 1) * sizeof(double));
        }


        // X[n_obj][dim]
        void fit(double** X, double* y, int n_obj, double lr, int n_epoch){
            double total_loss = 0;

            for(int k=0; k<n_epoch; ++k){
                
                std::cout << "Epoch " << k + 1 << "/ " << n_epoch;

                //computing y_pred - y
                double* pred = pred_batch(X, n_obj);
                double* diff = (double*)calloc(n_obj, sizeof(double));

                for(int i=0; i<n_obj; ++i){
                    diff[i] = pred[i] - y[i];
                }

                free(pred);



                // computing gradient
                double* grad = (double*)calloc((dim+1), sizeof(double));
                grad[dim] = 0;

                for(int j=0; j<n_obj; ++j){
                    grad[dim] += diff[j];
                }

                for(int i=0; i<dim; ++i){
                    grad[i] = 0;
                    for(int j=0; j<n_obj; ++j){
                        grad[i] += X[j][i] * diff[j];
                    }
                }

                for(int i=0; i<dim; ++i){
                    grad[i] *= (2.0 / n_obj);
                }



                // computing MSE
                double epoch_loss = 0;

                for(int j=0; j<n_obj; ++j){
                    epoch_loss += diff[j]*diff[j];
                }

                epoch_loss /= n_obj;
                free(diff);



                //applying gradient
                for(int i=0; i<dim; ++i){
                    weights[i] -= lr * grad[i];
                }

                free(grad);

                std::cout << "  |  epoch_loss " << epoch_loss << "\n";
                total_loss += epoch_loss;
            }

            std::cout << "\ntotal_mean_loss " << total_loss / n_epoch << "\n";
        }


        ~LinReg(){
            free(weights);
        }

};








int main(){

    std::ifstream file("./data/data.txt");
    int n_obj;
    int dim;
    file >> n_obj;
    file >> dim;
    
    double** X = (double**)calloc(n_obj, sizeof(double*));
    for(int i=0; i<n_obj; ++i){
        X[i] = (double*)calloc(dim, sizeof(double));
    }

    double* y = (double*)calloc(n_obj, sizeof(double));


    for(int i=0; i<n_obj; ++i){
        for(int j=0; j<dim; ++j){
            file >> X[i][j];
        }
        file >> y[i];
    }

    file.close();


    LinReg LR1(dim);

    LR1.fit(X, y, n_obj, 1.0, 8);
    

}