#ifndef LINREG_H
#define LINREG_H


#include <iostream>
#include <vector>
#include <fstream>
#include <tuple>
#include <limits>
#include <cstring>
#include <cmath>


double dot(double* x1, double* x2, int dim);


class LinReg{
    private:
        
        int dim;
        double* weights;
        int total_epochs = 0;

    public:

        LinReg(int d, double a, double b);

    
        // x[dim]
        double pred_single(double* x) const;


        // X[n_obj][dim]
        double* pred_batch(double** X, int n_obj) const;


        // return w from R^(dim+1) where w_0 = w[dim] = bias term
        double* GetWeights() const;


        // set weights to w where w[dim] = w_0 = bias term
        void SetWeights(double *w);


        // X[n_obj][dim]
        void fit(double** X, double* y, int n_obj, double initial_lr, double lr_decay, double min_lr, int n_epoch, bool verbose);


        friend std::ostream& operator << (std::ostream& out, const LinReg& LR);


        ~LinReg();

};


std::tuple<double**, double*, int, int> process_data(int max_obj);


void free_data(double** X, double* y, int n_obj);


class LRSchedulerBase{
    public:

        int _n_epochs;
        double _initial_lr;
        double _decay_rate;
        double _curr_lr;

        LRSchedulerBase(int n_epochs, double initial_lr, double decay_rate );

        virtual double Step(int epoch) = 0;

        // virtual ~LRSchedulerBase();
};



class StepDecay : public LRSchedulerBase{
    private:
        int _step_size;
    public:
        StepDecay(int n_epochs, double initial_lr, double decay_rate, int step_size );
            
        double Step(int epoch) override;
};




class ExponentialDecay : public LRSchedulerBase{
    public:
        ExponentialDecay(int n_epochs, double initial_lr, double decay_rate);
        
        double Step(int epoch) override;
};


class CosineDecay : public LRSchedulerBase{
    private:
        double _min_lr;
        int _T;
    public:
        CosineDecay(int n_epochs, double initial_lr, double min_lr, int T);
            
        double Step(int epoch) override;
};

#endif // LINREG_H