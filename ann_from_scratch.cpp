#include <iostream>
#include <vector>
#include <random>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <string>
#include <math.h>
#include "eigen-3.3.7/Eigen/StdVector"
#include "eigen-3.3.7/Eigen/Dense"
#include "eigen-3.3.7/Eigen/Core"

using namespace std;
using namespace Eigen;

struct struct_parameters{
  vector <MatrixXd> weights;
  vector <MatrixXd> biases;
} ;

struct struct_linear_forward{
  MatrixXd Z;
  vector <MatrixXd> cache;
} ;

struct struct_relu_function{
  MatrixXd A;
  MatrixXd cache;
} ;

struct struct_sigmoid_function{
  MatrixXd A;
  MatrixXd cache;
} ;

struct struct_propagation{
  MatrixXd A;
  vector < vector <MatrixXd> > caches;
} ;

struct d_parameters{
  MatrixXd dW;
  MatrixXd db;
  MatrixXd dA;
} ;

struct gradients{
  vector <MatrixXd> dW;
  vector <MatrixXd> db;
} ;

struct dataset{
  vector <MatrixXd> X_train;
  vector <MatrixXd> Y_train;
  vector <MatrixXd> X_test;
  vector <MatrixXd> Y_test;
} ;

struct csv_dataset{
  MatrixXd X;
  MatrixXd Y;
} ;

struct splitted_datasets{
  MatrixXd X_train;
  MatrixXd Y_train;
  MatrixXd X_test;
  MatrixXd Y_test;
} ;

struct adam_parameters{
  vector <MatrixXd> v_weights;
  vector <MatrixXd> v_biases;
  vector <MatrixXd> s_weights;
  vector <MatrixXd> s_biases;
  struct_parameters parameters;
} ;

struct adam_corrected{
  vector <MatrixXd> v_corrected_weights;
  vector <MatrixXd> v_corrected_biases;
  vector <MatrixXd> s_corrected_weights;
  vector <MatrixXd> s_corrected_biases;
} ;

struct minibatches{
  vector <MatrixXd> X_train;
  vector <MatrixXd> Y_train;
} ;

struct_sigmoid_function sigmoid(ArrayXXd z);
struct_relu_function relu(ArrayXXd z);
struct_parameters initialize_parameters(int size, int layers_dims[], int seed);
struct_linear_forward linear_forward_propagation(MatrixXd A_prev, MatrixXd weights, MatrixXd biases);
struct_propagation forward_propagation(MatrixXd X, vector <MatrixXd> weights, vector <MatrixXd> biases, int layers_dims[], int size);
double compute_cost(MatrixXd y_pred, MatrixXd Y);
d_parameters linear_backward(MatrixXd dZ);
MatrixXd relu_backward(MatrixXd dA, MatrixXd A);
gradients backpropagation(MatrixXd Y, MatrixXd y_pred, vector <vector <MatrixXd> > caches);
struct_parameters update_parameters(struct_parameters parameters, vector <MatrixXd> dW, vector <MatrixXd> db, float learning_rate);
struct_parameters model_sgd(vector <MatrixXd> X, vector <MatrixXd> Y, int size, int layers_dims[], int num_epochs, float learning_rate, int seed);
dataset build_dataset(MatrixXd X, MatrixXd Y, int split_size);
csv_dataset read_csv(string filename);
VectorXd standard_deviation(MatrixXd X, VectorXd mean_X);
MatrixXd normalize_dataset(MatrixXd X);
vector <double> predict(MatrixXd X_test, vector <MatrixXd> parameters, vector <MatrixXd> biases, int layers_dims[], int size);
double compute_accuracy(vector <double> predictions, MatrixXd Y_test);
struct_parameters model_batch_gd(MatrixXd X_train, MatrixXd Y_train, int size, int layers_dims[], int num_epochs, float learning_rate, int seed);
splitted_datasets split_dataset(MatrixXd X, MatrixXd Y, float split_size);
adam_parameters initialize_adam(struct_parameters parameters);
adam_parameters adam_optimizer(adam_parameters adam_parameters, vector <MatrixXd> dW, vector <MatrixXd> db, int t, float learning_rate, double beta1, double beta2, double epsilon);
minibatches random_minibatches(MatrixXd X_train, MatrixXd Y_train, int minibatch_size);
struct_parameters model_minibatches(MatrixXd X_train, MatrixXd Y_train, int size, int layers_dims[], int num_epochs, float learning_rate, int seed, int minibatch_size);

int main(){

    int layers_dims[4] = {8, 10, 10, 1}, seed, num_epochs = 2, X_train_size, size = 4, minibatch_size = 32;
    float learning_rate = 0.1, split_size = 0.9;
    double accuracy, global_accuracy = 0;

    cout << "Enter the seed: ";
    cin >> seed;

    csv_dataset csv_dataset;
    struct_parameters parameters;
    dataset dataset;
    splitted_datasets splitted_datasets;
    MatrixXd X_train_csv;
    MatrixXd Y_train_csv;
    vector <MatrixXd> X_train;
    vector <MatrixXd> Y_train;
    vector <MatrixXd> X_test;
    vector <MatrixXd> Y_test;
    vector <double> predictions;
    minibatches minibatches;

    csv_dataset = read_csv("pulsar_stars.csv");

    X_train_csv = csv_dataset.X;
    Y_train_csv = csv_dataset.Y;

    // ----------------------------------- STOCHASTIC GRADIENT DESCENT -------------------------------------------------------------------------------------------------

    // PREPROCESSING
    X_train_csv = normalize_dataset(X_train_csv);
    X_train_size = (int)round(split_size * X_train_csv.cols());
    dataset = build_dataset(X_train_csv, Y_train_csv, X_train_size);

    // MODEL
    parameters = model_sgd(dataset.X_train, dataset.Y_train, size, layers_dims, num_epochs, learning_rate, seed);

    // PREDICTIONS AND ACCURACY
    MatrixXd matrix_y_test = MatrixXd :: Zero(dataset.Y_test[0].cols(), dataset.Y_test.size());
    vector <double> temp_prediction;
    for(int j = 0; j < dataset.Y_test.size(); j++){
      matrix_y_test(0, j) = dataset.Y_test[j](0, 0);
    }
    for(int i = 0; i < dataset.X_test.size(); i++){
      temp_prediction = predict(dataset.X_test[i], parameters.weights, parameters.biases, layers_dims, size);
      predictions.push_back(temp_prediction[0]);
    }
    accuracy = compute_accuracy(predictions, matrix_y_test);

    cout << "Test Accuracy: " << accuracy << endl;

    cout << endl;

    // --------------------------------------- END ------------------------------------------------------------------------------------------------------------------------------

    // ----------------------------------- BATCH GRADIENT DESCENT ----------------------------------------------------------------------------------------------------------------

    // PREPROCESSING
    X_train_csv = normalize_dataset(X_train_csv);
    splitted_datasets = split_dataset(X_train_csv, Y_train_csv, split_size);

    // MODEL
    parameters = model_batch_gd(splitted_datasets.X_train, splitted_datasets.Y_train, size, layers_dims, num_epochs, learning_rate, seed);

    // PREDICTIONS AND ACCURACY
    predictions = predict(splitted_datasets.X_test, parameters.weights, parameters.biases, layers_dims, size);
    accuracy = compute_accuracy(predictions, splitted_datasets.Y_test);

    cout << "Test Accuracy: " << accuracy << endl;

    cout << endl;

    // ------------------------------------------ END ------------------------------------------------------------------------------------------------------------------------------------

    // ----------------------------------------- MINIBATCH GRADIENT DESCENT --------------------------------------------------------------------------------------------------------------

    // PREPROCESSING
    X_train_csv = normalize_dataset(X_train_csv);
    splitted_datasets = split_dataset(X_train_csv, Y_train_csv, split_size);

    // MODEL
    parameters = model_minibatches(splitted_datasets.X_train, splitted_datasets.Y_train, size, layers_dims, num_epochs, learning_rate, seed, minibatch_size);

    // PREDICTIONS AND ACCURACY
    predictions = predict(splitted_datasets.X_test, parameters.weights, parameters.biases, layers_dims, size);
    accuracy = compute_accuracy(predictions, splitted_datasets.Y_test);

    cout << "Test Accuracy: " << accuracy << endl;

    cout << endl;

    // --------------------------------------------- END ----------------------------------------------------------------------------------------------------------------------------------

  return 0;
}

struct_parameters initialize_parameters(int size, int layers_dims[], int seed){
  struct_parameters p;
  MatrixXd temp_parameters;
  MatrixXd temp_biases;
  default_random_engine generator;
  generator.seed(seed);
  normal_distribution <double> distribution(0.0, 1.0);

  for (int i = 1; i < size; i++){
      temp_parameters = MatrixXd :: Zero(layers_dims[i], layers_dims[i - 1]);
      temp_biases = MatrixXd :: Zero(layers_dims[i], 1);
      for (int j = 0; j < temp_parameters.rows(); j++){
          for (int z = 0; z < temp_parameters.cols(); z++)
              temp_parameters(j, z) = distribution(generator);
              distribution.reset();
      }
      p.weights.push_back(temp_parameters);
      p.biases.push_back(temp_biases);
      temp_parameters.resize(0, 0);

  }

  return p;

}

struct_sigmoid_function sigmoid(ArrayXXd z){
   int z_rows = z.rows(), z_cols = z.cols();
   struct_sigmoid_function sig;

   sig.A = (1 / (1 + exp(-z)));
   sig.A = sig.A.matrix();
   sig.cache = z;

   return sig;
}

struct_relu_function relu(ArrayXXd z){
  ArrayXXd zeros_matrix = ArrayXXd :: Zero(z.rows(), z.cols());
  struct_relu_function rel_func;

  rel_func.A = z.max(zeros_matrix);
  rel_func.A = rel_func.A.matrix();
  rel_func.cache = z;

  return rel_func;
}

adam_parameters initialize_adam(struct_parameters parameters){
    int L = parameters.weights.size();
    adam_parameters adam_parameters;
    MatrixXd temp_matrix;

    for(int l = 0; l < L; l++){
        adam_parameters.v_weights.push_back(temp_matrix = MatrixXd :: Zero(parameters.weights[l].rows(), parameters.weights[l].cols()));
        adam_parameters.v_biases.push_back(temp_matrix = MatrixXd :: Zero(parameters.biases[l].rows(), parameters.biases[l].cols()));
        adam_parameters.s_weights.push_back(temp_matrix = MatrixXd :: Zero(parameters.weights[l].rows(), parameters.weights[l].cols()));
        adam_parameters.s_biases.push_back(temp_matrix = MatrixXd :: Zero(parameters.biases[l].rows(), parameters.biases[l].cols()));
    }
    adam_parameters.parameters = parameters;

    return adam_parameters;
}

struct_linear_forward linear_forward_propagation(MatrixXd A_prev, MatrixXd weights, MatrixXd biases){
  struct_linear_forward forward;
  MatrixXd bias_matrix = MatrixXd :: Zero(biases.rows(), A_prev.cols());

  for(int i = 0; i < bias_matrix.cols(); i++){
      for(int j = 0; j < bias_matrix.rows(); j++){
          bias_matrix(j, i) = biases(j, 0);
      }
  }

  forward.Z = weights * A_prev + bias_matrix;

  forward.cache.push_back(A_prev);
  forward.cache.push_back(weights);
  forward.cache.push_back(biases);

  return forward;
}

struct_propagation forward_propagation(MatrixXd X, vector <MatrixXd> weights, vector <MatrixXd> biases, int layers_dims[], int size){
    MatrixXd A_prev;
    MatrixXd A;
    MatrixXd forward_z;
    struct_propagation propagation;
    struct_linear_forward forward;
    struct_relu_function rel_func;
    struct_sigmoid_function sig;

    A = X;

    for(int l = 1; l < size - 1; l++){
        A_prev = A;
        forward = linear_forward_propagation(A_prev, weights[l - 1], biases[l - 1]);
        rel_func = relu(forward.Z.array());
        A = rel_func.A;
        propagation.caches.push_back(forward.cache);
        A_prev.resize(0, 0);
    }

    forward = linear_forward_propagation(A, weights[size - 2], biases[size - 2]);
    sig = sigmoid(forward.Z.array());
    propagation.A = sig.A;
    propagation.caches.push_back(forward.cache);

    A.resize(0, 0);

    return propagation;
}

double compute_cost(MatrixXd y_pred, MatrixXd Y){
    double m = Y.cols();
    double cost;
    MatrixXd log_y_pred;
    MatrixXd Y_log_y_pred;
    MatrixXd ones = MatrixXd :: Constant(Y.rows(), Y.cols(), 1.0);
    MatrixXd Y_1;
    MatrixXd log_1_y_pred;
    MatrixXd y_pred_1;
    MatrixXd temp_cost;

    log_y_pred = log(y_pred.array()).matrix().transpose();
    Y_log_y_pred = Y * log_y_pred;
    Y_1 = ones - Y;
    y_pred_1 = ones - y_pred;
    log_1_y_pred = log(y_pred_1.array() + 0.00000001).matrix().transpose();

    temp_cost = Y * log_y_pred + Y_1 * log_1_y_pred;
    cost = -1 / m * temp_cost(0, 0);

    return cost;
}

MatrixXd relu_backward(MatrixXd dA, MatrixXd A){
  MatrixXd dZ;

  for(int i = 0; i < A.rows(); i++){
    for(int j = 0; j < A.cols(); j++){
      if(A(i, j) > 0){
        A(i, j) = 1;
      }
      else{
        A(i, j) = 0;
      }
    }
  }

  dZ = dA.array() * A.array();

  return dZ;
}

d_parameters linear_backward(MatrixXd dZ, MatrixXd A, MatrixXd W){
  d_parameters d_parameters;
  double m = dZ.cols();

  d_parameters.dW = (1 / m * (dZ * A.transpose()).array()).matrix();
  d_parameters.db = (1 / m * dZ.rowwise().sum().array()).matrix();
  d_parameters.dA = W.transpose() * dZ;

  return d_parameters;
}

gradients backpropagation(MatrixXd Y, MatrixXd y_pred, vector <vector <MatrixXd>> caches){
  MatrixXd dZ;
  d_parameters d_parameters;
  gradients gradients;
  int size = caches.size() - 1;

  dZ = y_pred.array() - Y.array();

  for(int i = size; i >= 0; i--){
    d_parameters = linear_backward(dZ.matrix(), caches[i][0], caches[i][1]);
    dZ = relu_backward(d_parameters.dA, caches[i][0]);
    gradients.dW.push_back(d_parameters.dW);
    gradients.db.push_back(d_parameters.db);
  }

  reverse(gradients.dW.begin(), gradients.dW.end());
  reverse(gradients.db.begin(), gradients.db.end());

  return gradients;
}

struct_parameters update_parameters(struct_parameters parameters, vector <MatrixXd> dW, vector<MatrixXd> db, float learning_rate){
  MatrixXd temp_weights;
  MatrixXd temp_biases;
  int size = parameters.weights.size();

  for(int i = 0; i <= size - 1; i++){
    temp_weights = parameters.weights[i].array() - learning_rate * dW[i].array();
    parameters.weights[i] = temp_weights.matrix();
    temp_biases = parameters.biases[i].array() - learning_rate * db[i].array();
    parameters.biases[i] = temp_biases.matrix();
  }

  return parameters;
}

adam_parameters adam_optimizer(adam_parameters adam_parameters, vector <MatrixXd> dW, vector <MatrixXd> db, int t, float learning_rate, double beta1, double beta2, double epsilon){
  int L = adam_parameters.parameters.weights.size();
  adam_corrected adam_corrected;
  MatrixXd temp_weights;
  MatrixXd temp_biases;

  for(int l = 0; l < L; l++){
      adam_parameters.v_weights[l] = (beta1 * adam_parameters.v_weights[l].array()).matrix() + ((1 - beta1) * dW[l].array()).matrix();
      adam_parameters.v_biases[l] = (beta1 * adam_parameters.v_biases[l].array()).matrix() + ((1 -beta1) * db[l].array()).matrix();

      adam_corrected.v_corrected_weights.push_back((adam_parameters.v_weights[l].array() / (1 - pow(beta1, t))).matrix());
      adam_corrected.v_corrected_biases.push_back((adam_parameters.v_biases[l].array() / (1 - pow(beta1, t))).matrix());

      adam_parameters.s_weights[l] = (beta2 * adam_parameters.s_weights[l].array()).matrix() + ((1 - beta2) * pow(dW[l].array(), 2)).matrix();
      adam_parameters.s_biases[l] = (beta2 * adam_parameters.s_biases[l].array()).matrix() + ((1 - beta2) * pow(db[l].array(), 2)).matrix();

      adam_corrected.s_corrected_weights.push_back((adam_parameters.s_weights[l].array() / (1 - pow(beta2, t))).matrix());
      adam_corrected.s_corrected_biases.push_back((adam_parameters.s_biases[l].array() / (1 - pow(beta2, t))).matrix());

      temp_weights = adam_parameters.parameters.weights[l].array() - learning_rate * (adam_corrected.v_corrected_weights[l].array() / (sqrt(adam_corrected.s_corrected_weights[l].array()) + epsilon));
      adam_parameters.parameters.weights[l] = temp_weights.matrix();

      temp_biases = adam_parameters.parameters.biases[l].array() - learning_rate * (adam_corrected.v_corrected_biases[l].array() / (sqrt(adam_corrected.s_corrected_biases[l].array()) + epsilon));
      adam_parameters.parameters.biases[l] = temp_biases.matrix();
  }

  return adam_parameters;
}

minibatches random_minibatches(MatrixXd X, MatrixXd Y, int minibatch_size){
  minibatches minibatches;
  MatrixXd temp_X_train = MatrixXd :: Zero(X.rows(), minibatch_size);
  MatrixXd temp_Y_train = MatrixXd :: Zero(1, minibatch_size);
  int num_minibatches = (int)round(X.cols() / minibatch_size);
  for(int l = 0; l < num_minibatches; l++){
    for(int i = minibatch_size * l; i < minibatch_size * (l + 1); i++){
      for(int j = 0; j < X.rows(); j++){
        temp_X_train(j, i - minibatch_size * l) = X(j, i);
      }
      temp_Y_train(0, i - minibatch_size * l) = Y(0, i);
    }
    minibatches.X_train.push_back(temp_X_train);
    minibatches.Y_train.push_back(temp_Y_train);
  }

  if(X.cols() % minibatch_size != 0){
    MatrixXd temp_X_train = MatrixXd :: Zero(X.rows(), X.cols() - minibatch_size * num_minibatches);
    MatrixXd temp_Y_train = MatrixXd :: Zero(1, X.cols() - minibatch_size * num_minibatches);
    for(int i = minibatch_size * num_minibatches; i < X.cols(); i++){
      for(int j = 0; j < X.rows(); j++){
        temp_X_train(j, i - minibatch_size * num_minibatches) = X(j, i);
      }
      temp_Y_train(0, i - minibatch_size * num_minibatches) = Y(0, i);
    }
    minibatches.X_train.push_back(temp_X_train);
    minibatches.Y_train.push_back(temp_Y_train);
  }

  return minibatches;
}

dataset build_dataset(MatrixXd X, MatrixXd Y, int split_size){
    MatrixXd temp_X;
    MatrixXd temp_Y;
    dataset dataset;

    for(int col = 0; col < X.cols(); col++){
      MatrixXd temp_X = MatrixXd :: Zero(X.rows(), 1);
      MatrixXd temp_Y = MatrixXd :: Zero(1, 1);
        for(int row = 0; row < X.rows(); row++){
            temp_X(row, 0) = X(row, col);
        }
        temp_Y(0, 0) = Y(0, col);

        dataset.X_train.push_back(temp_X);
        dataset.Y_train.push_back(temp_Y);
    }

    auto test_start_X = dataset.X_train.begin() + split_size + 1;
    auto test_end_X = dataset.X_train.end();
    auto train_start_X = dataset.X_train.begin();
    auto train_end_X = dataset.X_train.begin() + split_size;

    auto test_start_Y = dataset.Y_train.begin() + split_size + 1;
    auto test_end_Y = dataset.Y_train.end();
    auto train_start_Y = dataset.Y_train.begin();
    auto train_end_Y = dataset.Y_train.begin() + split_size;

    vector <MatrixXd> X_test(dataset.X_train.size() - split_size - 1);
    vector <MatrixXd> Y_test(dataset.Y_train.size() - split_size - 1);

    copy(test_start_Y, test_end_Y, Y_test.begin());
    copy(test_start_X, test_end_X, X_test.begin());
    dataset.X_test = X_test;
    dataset.Y_test = Y_test;

    vector <MatrixXd> X_train(split_size);
    vector <MatrixXd> Y_train(split_size);

    copy(train_start_X, train_end_X, X_train.begin());
    copy(train_start_Y, train_end_Y, Y_train.begin());
    dataset.X_train = X_train;
    dataset.Y_train = Y_train;

    return dataset;
}

struct_parameters model_sgd(vector <MatrixXd> X_train, vector <MatrixXd> Y_train, int size, int layers_dims[], int num_epochs, float learning_rate, int seed){
  struct_parameters parameters;
  struct_propagation propagation;
  gradients gradients;
  double cost, epoch_cost;
  float accuracy;
  vector <double> predictions;

  parameters = initialize_parameters(size, layers_dims, seed);

  for(int epoch = 1; epoch <= num_epochs; epoch++){
      epoch_cost = 0.0;
      for(int example = 0; example < X_train.size(); example++){
          propagation = forward_propagation(X_train[example], parameters.weights, parameters.biases, layers_dims, size);
          cost = compute_cost(propagation.A, Y_train[example]);
          gradients = backpropagation(Y_train[example], propagation.A, propagation.caches);
          parameters = update_parameters(parameters, gradients.dW, gradients.db, learning_rate);
          epoch_cost += cost;

      }
      epoch_cost /= X_train.size();
      cout << "Cost at epoch " << epoch << " is " << epoch_cost << endl;
  }
  MatrixXd matrix_y_train = MatrixXd :: Zero(Y_train[0].cols(), Y_train.size());
  vector <double> temp_prediction;

  for(int j = 0; j < Y_train.size(); j++){
      matrix_y_train(0, j) = Y_train[j](0, 0);
  }
  for(int i = 0; i < X_train.size(); i++){
    temp_prediction = predict(X_train[i], parameters.weights, parameters.biases, layers_dims, size);
    predictions.push_back(temp_prediction[0]);
  }
  accuracy = compute_accuracy(predictions, matrix_y_train);

  cout << endl;
  cout << "Training accuracy: " << accuracy << endl;

  return parameters;
}

struct_parameters model_batch_gd(MatrixXd X_train, MatrixXd Y_train, int size, int layers_dims[], int num_epochs, float learning_rate, int seed){
  struct_parameters parameters;
  struct_propagation propagation;
  gradients gradients;
  double cost;
  float accuracy;
  vector <double> predictions;

  parameters = initialize_parameters(size, layers_dims, seed);

  for(int epoch = 1; epoch <= num_epochs; epoch++){
      propagation = forward_propagation(X_train, parameters.weights, parameters.biases, layers_dims, size);
      cost = compute_cost(propagation.A, Y_train);
      gradients = backpropagation(Y_train, propagation.A, propagation.caches);
      parameters = update_parameters(parameters, gradients.dW, gradients.db, learning_rate);
      cout << "Cost at epoch " << epoch << " is " << cost << endl;
      }
  predictions = predict(X_train, parameters.weights, parameters.biases, layers_dims, size);
  accuracy = compute_accuracy(predictions, Y_train);
  cout << endl;
  cout << "Training accuracy: " << accuracy << endl;

  return parameters;
}

struct_parameters model_minibatches(MatrixXd X_train, MatrixXd Y_train, int size, int layers_dims[], int num_epochs, float learning_rate, int seed, int minibatch_size){
  struct_parameters parameters;
  struct_propagation propagation;
  adam_parameters adam_parameters;
  gradients gradients;
  minibatches minibatches;
  double cost;
  float accuracy;
  vector <double> predictions;

  double beta1 = 0.9, beta2 = 0.999, epsilon = 0.00000001;

  parameters = initialize_parameters(size, layers_dims, seed);
  adam_parameters = initialize_adam(parameters);

  for(int epoch = 1; epoch <= num_epochs; epoch++){
      double epoch_cost = 0.0;
      minibatches = random_minibatches(X_train, Y_train, minibatch_size);
      double num_minibatches = minibatches.X_train.size();
      int t = 0;
      for(int minibatch = 0; minibatch < num_minibatches; minibatch++){
          propagation = forward_propagation(minibatches.X_train[minibatch], adam_parameters.parameters.weights, adam_parameters.parameters.biases, layers_dims, size);
          cost = compute_cost(propagation.A, minibatches.Y_train[minibatch]);
          t++;
          gradients = backpropagation(minibatches.Y_train[minibatch], propagation.A, propagation.caches);
          adam_parameters = adam_optimizer(adam_parameters, gradients.dW, gradients.db, t, learning_rate, beta1, beta2, epsilon);
          epoch_cost += cost;
      }
      cout << "Cost at epoch " << epoch << " is " << epoch_cost / num_minibatches << endl;
  }
  predictions = predict(X_train, adam_parameters.parameters.weights, adam_parameters.parameters.biases, layers_dims, size);
  accuracy = compute_accuracy(predictions, Y_train);
  cout << endl;
  cout << "Training accuracy: " << accuracy << endl;

  parameters.weights = adam_parameters.parameters.weights;
  parameters.biases = adam_parameters.parameters.biases;

  return parameters;
}


csv_dataset read_csv(string filename){
  fstream fin;
  fin.open(filename, ios :: in);

  int i = 0, j = 0;
  string word, line;
  csv_dataset csv_dataset;
  csv_dataset.X = MatrixXd :: Zero(17898, 8);
  csv_dataset.Y = MatrixXd :: Zero(17898, 1);

  while(getline(fin, line)){
    stringstream s(line);
    while(getline(s, word, ',')){
      if(j == 8){
        csv_dataset.Y(i, 0) = stod(word);
        j = 0;
      }
      else
        {
          csv_dataset.X(i, j) = stod(word);
          j++;
        }
    }
    i++;
  }

  csv_dataset.X.transposeInPlace();
  csv_dataset.Y.transposeInPlace();

  return csv_dataset;
}

VectorXd standard_deviation(MatrixXd X, VectorXd mean_X){
  VectorXd stddev = VectorXd :: Zero(mean_X.rows(), mean_X.cols());
  double variation;

  for(int i = 0; i < X.rows(); i++){
      for(int j = 0; j < X.cols(); j++){
          variation += pow(X(i, j) - mean_X(i, 0), 2);
    }
      variation /= X.cols();
      stddev(i, 0) = sqrt(variation);

  }

  return stddev;
}

MatrixXd normalize_dataset(MatrixXd X){
    VectorXd mean_X;
    VectorXd stddev;

    mean_X = X.rowwise().mean();
    stddev = standard_deviation(X, mean_X);

    X.colwise() -= mean_X;
    X.array().colwise() /= stddev.array();

    return X.matrix();
}

vector <double> predict(MatrixXd X_test, vector <MatrixXd> weights, vector <MatrixXd> biases, int layers_dims[], int size){
  double epoch_cost = 0.0;
  vector <double> predictions;
  struct_propagation propagation;

  propagation = forward_propagation(X_test, weights, biases, layers_dims, size);
  for(int i = 0; i < propagation.A.cols(); i++){
    if (propagation.A(0, i) > 0.5){
        predictions.push_back(1);
    }
    else{
        predictions.push_back(0);
    }
  }

  return predictions;
}

double compute_accuracy(vector <double> predictions, MatrixXd Y_test){
    double cnt = 0;
    double accuracy;

    for(int i = 0; i < predictions.size(); i++){
        if(predictions[i] == Y_test(0, i)){
          cnt++;
        }
    }
    accuracy = cnt / Y_test.cols();

    return accuracy * 100;
}

splitted_datasets split_dataset(MatrixXd X, MatrixXd Y, float split_size){

    int training_set_size = (int)round(split_size * X.cols());
    int test_set_size = (int)round(X.cols() - training_set_size);
    splitted_datasets splitted_datasets;
    splitted_datasets.X_train = MatrixXd :: Zero(X.rows(), training_set_size);
    splitted_datasets.Y_train = MatrixXd :: Zero(Y.rows(), training_set_size);
    splitted_datasets.X_test = MatrixXd :: Zero(X.rows(), test_set_size);
    splitted_datasets.Y_test = MatrixXd :: Zero(Y.rows(), test_set_size);

    for(int i = 0; i < training_set_size; i++){
        for(int j = 0; j < X.rows(); j++){
            splitted_datasets.X_train(j, i) = X(j, i);
        }
        splitted_datasets.Y_train(0, i) = Y(0, i);
    }

    for(int i = 0; i < test_set_size; i++){
        for(int j = 0; j < X.rows(); j++){
            splitted_datasets.X_test(j, i) = X(j, i + training_set_size);
        }
        splitted_datasets.Y_test(0, i) = Y(0, i + training_set_size);
    }

    return splitted_datasets;
}
