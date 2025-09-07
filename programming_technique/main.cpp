#include <iostream>
#include <vector>
#include <cstring>
#include <random>
#include <fstream>
#include <tuple>


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
        int total_epochs = 0;

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
        void fit(double** X, double* y, int n_obj, double lr, int n_epoch, bool verbose){
            double total_loss = 0;
            double min_loss = std::numeric_limits<double>::max();
            double max_loss = 0;

            for(int k=0; k<n_epoch; ++k){
                ++total_epochs;
                if(verbose){
                    std::cout << "Epoch " << k + 1 << " / " << n_epoch;
                }
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

                //applying gradient
                for(int i=0; i<dim; ++i){
                    weights[i] -= lr * grad[i];
                }

                free(grad);
                
                // computing MSE
                double epoch_loss = 0;

                for(int j=0; j<n_obj; ++j){
                    epoch_loss += diff[j]*diff[j];
                }

                epoch_loss /= n_obj;
                min_loss = std::min(epoch_loss, min_loss);
                max_loss = std::max(epoch_loss, max_loss);
                free(diff);



                if(verbose){
                    std::cout << "  |  epoch_loss " << epoch_loss << "\n";
                }
                total_loss += epoch_loss;
            }

            if(verbose){
                std::cout << "\ntotal_mean_loss " << total_loss / n_epoch << "\n";
                std::cout << "min_loss " << min_loss << "\n";
                std::cout << "max_loss " << max_loss << "\n";
            }
        }
        

        friend std::ostream& operator << (std::ostream& out, const LinReg& LR){
            for(int i=0; i<64; ++i){
                out<<'*';
            }

            out << "\nLinearRegression model\n";
            out << "dim = " << LR.dim <<"\n";
            out << "was fitted " << LR.total_epochs <<" times\n";
            for(int i=0; i<64; ++i){
                out<<'*';
            }
            out<<'\n';

            return out;
        }

        ~LinReg(){
            free(weights);
        }

};


std::tuple<double**, double*, int, int> process_data(int max_obj){

    // data reading
    std::ifstream file("./data/data.txt");
    int n_obj;
    int dim;
    file >> n_obj;
    file >> dim;
    n_obj = std::min(n_obj, max_obj);
    double** X = (double**)calloc(n_obj, sizeof(double*));

    for(int i=0; i<n_obj; ++i){
        X[i] = (double*)calloc(dim, sizeof(double));
    }

    double* y = (double*)calloc(n_obj, sizeof(double));
    double* max_val = (double*)calloc(dim+1, sizeof(double));

    for(int i=0; i<n_obj; ++i){
        for(int j=0; j<dim; ++j){
            file >> X[i][j];
            max_val[j] = std::max(max_val[j], std::abs(X[i][j]));
        }
        file >> y[i];
        max_val[dim] = std::max(max_val[dim], std::abs(y[i]));
    }

    file.close();


    // data normalizing  (min_max)
    for(int i=0; i<n_obj; ++i){
        for(int j=0; j<dim; ++j){
            X[i][j] /= max_val[j];
        }
        y[i] /= max_val[dim];
    }

    free(max_val);

    std::tuple<double**, double*, int, int> result = {X,y,dim,n_obj};
    return result;
}


void free_data(double** X, double* y, int n_obj){

    for(int i=0; i<n_obj; ++i){
        free(X[i]);
    }
    free(y);
    free(X);
}



int main(){

    auto [X, y, dim, n_obj] = process_data(512);

    LinReg LR1(dim);

    std::cout << LR1;
    LR1.fit(X, y, n_obj, 1.0, 128, false);
    std::cout << LR1;

    free_data(X, y, n_obj);
}


