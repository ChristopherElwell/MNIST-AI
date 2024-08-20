#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <array>
#include <cmath>
#include <random>
#include <iomanip>
#include <chrono>
#include <thread>
#include <Eigen/Dense>
#include <filesystem>


using namespace std;
using namespace Eigen;
constexpr int hidden_sizes[] = {32,32};
constexpr int input_size = 28*28;
constexpr int num_layers = 2; 
constexpr int output_size = 10;
constexpr float INIITAL_LEARNING_RATE = 0.01;
constexpr float DECAY_RATE = 0.5;
constexpr float alpha = 0.01f;
constexpr int EPOCH_SIZE = 60000;
    

// sigmoid
float sigmoid(float n){
    return 1.0f / (1.0f + exp(-n));
}

float sigmoid_prime(float n){
    return sigmoid(n) * (1.0f - sigmoid(n));
}

template <int length>
Matrix<float,length,1> vec_sigmoid(const Matrix<float,length,1>& vec){
    return vec.unaryExpr([](float item) -> float {return sigmoid(item);});
}

template <int length>
Matrix<float,length,1> vec_sigmoid_prime(const Matrix<float,length,1>& vec){
    return vec.unaryExpr([](float item) -> float {return sigmoid_prime(item);});
}

// ReLU
float ReLU(float n){
    return max(0.0f,n);
}

float ReLU_Prime(float n){
    return n > 0 ? 1.0f : 0.0f;
}

template <int length>
Matrix<float,length,1> vec_ReLU(const Matrix<float,length,1>& vec){
    return vec.unaryExpr([](float item) -> float {return ReLU(item);});
}

template <int length>
Matrix<float,length,1> vec_ReLU_prime(const Matrix<float,length,1>& vec){
    return vec.unaryExpr([](float item) -> float {return ReLU_Prime(item);});
}

// Leaky ReLU
float leakyReLU(float n){
    return n > 0 ? n : alpha * n;
}

float leakyReLU_Prime(float n){
    return n > 0 ? 1.0f : alpha;
}

template <int length>
Matrix<float,length,1> vec_leakyReLU(const Matrix<float,length,1>& vec){
    return vec.unaryExpr([](float item) -> float {return leakyReLU(item);});
}

template <int length>
Matrix<float,length,1> vec_leakyReLU_prime(const Matrix<float,length,1>& vec){
    return vec.unaryExpr([](float item) -> float {return leakyReLU_Prime(item);});
}

// softMax
template <int length>
Matrix<float,length,1> softMax(const Matrix<float,length,1>& vec){
    Matrix<float,length,1> stabilized = vec.array() - vec.maxCoeff();
    Matrix<float,length,1> exp = stabilized.array().exp();
    return exp / exp.sum();
}

struct Network {
    public:
        // weights
        Matrix<float,hidden_sizes[0],input_size> w0;
        Matrix<float,hidden_sizes[1],hidden_sizes[0]> w1;
        Matrix<float,output_size,hidden_sizes[1]> w2;

        // biases
        Matrix<float,hidden_sizes[0],1> b0;
        Matrix<float,hidden_sizes[1],1> b1;
        Matrix<float,output_size,1> b2;

        

    void HeDist(){
        random_device rd;
        mt19937 gen(rd());

        float mean = 0.0f;

        normal_distribution<float> disw0(mean,sqrt(2.0f / input_size));
        for (int j = 0; j < w0.rows(); j++){
            for (int i = 0; i < w0.cols(); i++){
                w0(j,i) = disw0(gen);
            }
        }
        normal_distribution<float> disw1(mean,sqrt(2.0f / hidden_sizes[0]));
        for (int j = 0; j < w1.rows(); j++){
            for (int i = 0; i < w1.cols(); i++){
                w1(j,i) = disw1(gen);
            }
        }
        normal_distribution<float> disw2(mean,sqrt(2.0f / hidden_sizes[1]));
        for (int j = 0; j < w2.rows(); j++){
            for (int i = 0; i < w2.cols(); i++){
                w2(j,i) = disw2(gen);
            }
        }
        
        b0.setZero();
        b1.setZero();
        b2.setZero();

    }

    void Zero(){
        w0.setZero();
        w1.setZero();
        w2.setZero();

        b0.setZero();
        b1.setZero();
        b2.setZero();
    } 

    void fromCSV(string file_name){
        ifstream file(file_name);

        string line;
        string cell;
        int j;


        while(getline(file,line)){
            
            j = 0;
            while(getline(file,line)){
                if (line.substr(0,5) == "BEGIN") break;
                stringstream lineStream(line);
                int i = 0;
                while(getline(lineStream, cell, ',')){
                    w0(j,i) = stof(cell);
                    i++;
                }
                j++;
            }
            j = 0;
            while(getline(file,line)){
                if (line.substr(0,5) == "BEGIN") break;
                stringstream lineStream(line);
                int i = 0;
                while(getline(lineStream, cell, ',')){
                    w1(j,i) = stof(cell);
                    i++;
                }
                j++;
            }
            j = 0;
            while(getline(file,line)){
                if (line.substr(0,5) == "BEGIN") break;
                stringstream lineStream(line);
                int i = 0;
                while(getline(lineStream, cell, ',')){
                    w2(j,i) = stof(cell);
                    i++;
                }
                j++;
            }
            j = 0;
            while(getline(file,line)){
                if (line.substr(0,5) == "BEGIN") break;
                stringstream lineStream(line);
                int i = 0;
                while(getline(lineStream, cell, ',')){
                    b0(j,i) = stof(cell);
                    i++;
                }
                j++;
            }
            j = 0;
            while(getline(file,line)){
                if (line.substr(0,5) == "BEGIN") break;
                stringstream lineStream(line);
                int i = 0;
                while(getline(lineStream, cell, ',')){
                    b1(j,i) = stof(cell);
                    i++;
                }
                j++;
            }
            j = 0;
            while(getline(file,line)){
                if (line.substr(0,5) == "BEGIN") break;
                stringstream lineStream(line);
                int i = 0;
                while(getline(lineStream, cell, ',')){
                    b2(j,i) = stof(cell);
                    i++;
                }
                j++;
            }

            
        }
    }
};

// not written by me
int swapEndianness(int num) {
    return ((num >> 24) & 0x000000FF) |
        ((num >> 8)  & 0x0000FF00) |
        ((num << 8)  & 0x00FF0000) |
        ((num << 24) & 0xFF000000);
}

void readImages(vector<Matrix<float,input_size,1>>& images){
    string images_name = "train-images.idx3-ubyte";
    ifstream file (filesystem::absolute(__FILE__).parent_path().string() + "\\archive\\" + images_name, ios::binary);
    
    if (file.is_open()){
        int magic, numImages, rows, cols;

        file.read((char*)&magic,sizeof(magic)); 
        file.read((char*)&numImages,sizeof(numImages)); 
        file.read((char*)&rows,sizeof(rows)); 
        file.read((char*)&cols,sizeof(cols)); 

        magic = swapEndianness(magic);
        numImages = swapEndianness(numImages);
        rows = swapEndianness(rows);
        cols = swapEndianness(cols);

        images.resize(numImages);
        for (int n = 0; n < numImages; n++){
            vector<uint8_t> buffer(rows * cols);
            file.read(reinterpret_cast<char*>(buffer.data()), input_size);

            for (int i = 0; i < input_size; i++){
                images[n](i) = static_cast<float>(buffer[i]) / 255.0f;
            }
        }
    }

    file.close();
}

void readLabels(vector<uint8_t>& labels){
    string labels_name = "train-labels.idx1-ubyte";
    ifstream file (filesystem::absolute(__FILE__).parent_path().string() + "\\archive\\" + labels_name, ios::binary);
    
    if (file.is_open()){
        int magic, numLabels;

        file.read((char*)&magic,sizeof(magic)); 
        file.read((char*)&numLabels,sizeof(numLabels)); 
        

        magic = swapEndianness(magic);
        numLabels = swapEndianness(numLabels);

        labels.resize(numLabels);

        file.read(reinterpret_cast<char*>(labels.data()), numLabels);

        file.close();
    }
}

Matrix<float, output_size, 1> neural(const Matrix<float, input_size, 1>& input, const Network net){
    
    Matrix<float, hidden_sizes[0], 1> z0 = net.w0*input + net.b0;
    Matrix<float, hidden_sizes[0], 1> a0 = vec_leakyReLU(z0);

    Matrix<float, hidden_sizes[1], 1> z1 = net.w1*a0 + net.b1;
    Matrix<float, hidden_sizes[1], 1> a1 = vec_leakyReLU(z1);
    
    Matrix<float, output_size, 1> z2 = net.w2*a1 + net.b2;
    return softMax(z2);
}

float get_cost(const Matrix<float, output_size, 1>& neural_out, const Matrix<float, output_size, 1>& actual_out){
    Matrix<float, output_size, 1> diff = actual_out - neural_out;
    return diff.squaredNorm();
}

bool verify(const Matrix<float, output_size, 1>& neural_out, int answer){
    int row, col;
    neural_out.maxCoeff(&row, &col);
    return answer == row;
}

void back_prop(const Matrix<float, input_size, 1>& input, int answer, const Network& net, Network& gradient){
    Matrix<float,output_size,1> y;
    y.setZero();
    y(answer) = 1.0f;

    // forward propagation
    Matrix<float, hidden_sizes[0], 1> z0 = net.w0*input + net.b0;
    Matrix<float, hidden_sizes[0], 1> a0 = vec_leakyReLU(z0);

    Matrix<float, hidden_sizes[1], 1> z1 = net.w1*a0 + net.b1;
    Matrix<float, hidden_sizes[1], 1> a1 = vec_leakyReLU(z1);
    
    Matrix<float, output_size, 1> z2 = net.w2*a1 + net.b2;
    Matrix<float, output_size, 1> a2 = softMax(z2);

    // backward propagation
    Matrix<float, output_size,1> loss2 = a2 - y;
    Matrix<float, hidden_sizes[1], 1> loss1 = (net.w2.transpose() * loss2).cwiseProduct(vec_leakyReLU_prime(z1));
    Matrix<float, hidden_sizes[0], 1> loss0 = (net.w1.transpose() * loss1).cwiseProduct(vec_leakyReLU_prime(z0));
    
    // nudges
    gradient.w0 += loss0 * input.transpose();
    gradient.w1 += loss1 * a0.transpose();
    gradient.w2 += loss2 * a1.transpose();

    gradient.b0 += loss0;
    gradient.b1 += loss1;
    gradient.b2 += loss2;
}

void update_net(Network& net, const Network& gradient, int N, float learningRate){
    net.w0 -= learningRate * gradient.w0 / N;
    net.w1 -= learningRate * gradient.w1 / N;
    net.w2 -= learningRate * gradient.w2 / N;
    
    net.b0 -= learningRate * gradient.b0 / N;
    net.b1 -= learningRate * gradient.b1 / N;
    net.b2 -= learningRate * gradient.b2 / N;
}

void draw_num(const Matrix<float,input_size,1>& imgs){
    cout << "\n\n\n";
    for (int i = 0; i < 58; i++){
        cout << "_";
    }
    cout << "\n|";
    for (int j = 0; j < 28; j++){
        for (int i = 0; i < 28; i++){
            int index = j * 28 + i;
            if (imgs(index) < 0.25) cout << "  ";
            else if (imgs(index) < 0.5) cout << "` ";
            else if (imgs(index) < 0.75) cout << "^ ";
            else cout << "# ";
        }
        cout << "|\n|";
    }
    for (int i = 0; i < 56; i++){
        cout << "_";
    }
    cout << "|";
}

Matrix<float,output_size,1> getOneHotEncoding(int answer){
    Matrix<float,output_size,1> answer_vec;
    answer_vec.setZero();
    answer_vec(answer) = 1.0f;
    return answer_vec;
}

bool gradientCheck(const Network& net, const Matrix<float, input_size, 1>& input, int answer) {
    constexpr float epsilon = 1e-2;
    Network gradientBP;
    gradientBP.Zero();
    back_prop(input, answer, net, gradientBP);

    // Check gradients for weights
    cout << "\n\n\nB0\n";
    for (int i = 0; i < net.b0.rows(); i++){
        Network netPlus = net;
        Network netMinus = net;
        netPlus.b0(i, 0) += epsilon;
        netMinus.b0(i, 0) -= epsilon;

        float costPlus = get_cost(neural(input, netPlus), getOneHotEncoding(answer));
        float costMinus = get_cost(neural(input, netMinus), getOneHotEncoding(answer));
        
        float numericalGradient = (costPlus - costMinus) / (2 * epsilon);
        float backpropGradient = gradientBP.b0(i, 0);

        cout << i << " NUMERICAL: " << numericalGradient << "  BACKPROP: " << backpropGradient << endl;
    }
    cout << "\n\n\nB1\n";
    for (int i = 0; i < net.b1.rows(); i++){
        Network netPlus = net;
        Network netMinus = net;
        netPlus.b1(i, 0) += epsilon;
        netMinus.b1(i, 0) -= epsilon;

        float costPlus = get_cost(neural(input, netPlus), getOneHotEncoding(answer));
        float costMinus = get_cost(neural(input, netMinus), getOneHotEncoding(answer));
        
        float numericalGradient = (costPlus - costMinus) / (2 * epsilon);
        float backpropGradient = gradientBP.b1(i, 0);

        cout << i << " NUMERICAL: " << numericalGradient << "  BACKPROP: " << backpropGradient << endl;
    }
    cout << "\n\n\nB2\n";
    for (int i = 0; i < net.b2.rows(); i++){
        Network netPlus = net;
        Network netMinus = net;
        netPlus.b2(i, 0) += epsilon;
        netMinus.b2(i, 0) -= epsilon;

        float costPlus = get_cost(neural(input, netPlus), getOneHotEncoding(answer));
        float costMinus = get_cost(neural(input, netMinus), getOneHotEncoding(answer));
        
        float numericalGradient = (costPlus - costMinus) / (2 * epsilon);
        float backpropGradient = gradientBP.b2(i, 0);

        cout << i << " NUMERICAL: " << numericalGradient << "  BACKPROP: " << backpropGradient << endl;
    }
    cout << "\n\n\nW0\n";
    for (int i = 0; i < net.w0.rows(); i++){
        for (int j = 0; j < net.w0.cols(); j++){
            Network netPlus = net;
            Network netMinus = net;
            netPlus.w0(i, j) += epsilon;
            netMinus.w0(i, j) -= epsilon;

            float costPlus = get_cost(neural(input, netPlus), getOneHotEncoding(answer));
            float costMinus = get_cost(neural(input, netMinus), getOneHotEncoding(answer));
            
            float numericalGradient = (costPlus - costMinus) / (2 * epsilon);
            float backpropGradient = gradientBP.w0(i, j);

            cout << i << ", " << j << " NUMERICAL: " << numericalGradient << "  BACKPROP: " << backpropGradient << endl;
        }
    }
    cout << "\n\n\nW1\n";
    for (int i = 0; i < net.w1.rows(); i++){
        for (int j = 0; j < net.w0.cols(); j++){
            Network netPlus = net;
            Network netMinus = net;
            netPlus.w1(i, j) += epsilon;
            netMinus.w1(i, j) -= epsilon;

            float costPlus = get_cost(neural(input, netPlus), getOneHotEncoding(answer));
            float costMinus = get_cost(neural(input, netMinus), getOneHotEncoding(answer));
            
            float numericalGradient = (costPlus - costMinus) / (2 * epsilon);
            float backpropGradient = gradientBP.w1(i, j);

            cout << i << ", " << j << " NUMERICAL: " << numericalGradient << "  BACKPROP: " << backpropGradient << endl;
        }
    }
    cout << "\n\n\nW2\n";
    for (int i = 0; i < net.w2.rows(); i++){
        for (int j = 0; j < net.w0.cols(); j++){
            Network netPlus = net;
            Network netMinus = net;
            netPlus.w2(i, j) += epsilon;
            netMinus.w2(i, j) -= epsilon;

            float costPlus = get_cost(neural(input, netPlus), getOneHotEncoding(answer));
            float costMinus = get_cost(neural(input, netMinus), getOneHotEncoding(answer));
            
            float numericalGradient = (costPlus - costMinus) / (2 * epsilon);
            float backpropGradient = gradientBP.w2(i, j);

            cout << i << ", " << j << " NUMERICAL: " << numericalGradient << "  BACKPROP: " << backpropGradient << endl;
        }
    }
    // Repeat for w1, w2, b0, b1, b2...

    return true;
}

void write_net_to_csv(const Network& net, string file_name){
    auto writeToFile = [](const auto& matrix, std::ofstream& file) {
        file << std::setprecision(7) << matrix.format(Eigen::IOFormat(
                Eigen::StreamPrecision, 
                Eigen::DontAlignCols,
                ", ", // Column separator
                "\n"  // Row separator
            ));
    };

    ofstream file(file_name);
    if (file.is_open()){
        file << "BEGIN w0\n";
        writeToFile(net.w0,file);
        file << "\nBEGIN w1\n";
        writeToFile(net.w1,file);
        file << "\nBEGIN w2\n";
        writeToFile(net.w2,file);
        file << "\nBEGIN b0\n";
        writeToFile(net.b0,file);
        file << "\nBEGIN b1\n";
        writeToFile(net.b1,file);
        file << "\nBEGIN b2\n";
        writeToFile(net.b2,file);
    }
}

void train(Network& net, vector<Matrix<float,input_size,1>>& images, vector<int>& labels,string save_dir){
    int NUM_EPOCHS = 50;
    vector<int> index;
    index.resize(EPOCH_SIZE);

    for (int i = 0; i < EPOCH_SIZE; i++){
        index[i] = i;
    }

    random_device rd;
    mt19937 g(rd());

    unsigned int total_cores = thread::hardware_concurrency();
    unsigned int cores_to_use = (total_cores * 3) / 4;
    if (cores_to_use < 1) cores_to_use = 1;

    int batch_size = 60;
    int props_per_core = batch_size / cores_to_use;

    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++){

        shuffle(index.begin(),index.end(),g);
        int data_count = 0;
        auto begin = chrono::system_clock::now();
        float sum = 0;
        int correct = 0;
        for (int i = 0; i < 1000; i++){
            sum += get_cost(neural(images[index[i]],net),getOneHotEncoding(labels[index[i]]));
            correct += verify(neural(images[index[i]],net),labels[index[i]]);
        }

        float learning_rate = INIITAL_LEARNING_RATE / (1.0f + epoch * DECAY_RATE);
        cout << "\n\nBeggining epoch: " << (epoch + 1) << endl;
        cout << "Current avg error: " << sum / 1000.0f << endl;
        cout << "Percent correct: " << correct / 10.0f << "%" << endl;
        cout << "Learning rate: " << learning_rate << endl;
        float learning = INIITAL_LEARNING_RATE / (1.0f + DECAY_RATE * data_count);
        
        while (data_count + batch_size < EPOCH_SIZE){
            vector<thread> threads;
            vector<Network> thread_gradients(cores_to_use, Network());
            for (int t = 0; t < cores_to_use; t++){

                threads.emplace_back([&thread_gradients, t, &net, &images, &labels, index, props_per_core, data_count]() {
                    thread_gradients[t].Zero();
                    for (int i = 0; i < props_per_core; i++){
                        int idx = data_count + i + props_per_core * t;
                        back_prop(images[index[idx]],labels[index[idx]],net,thread_gradients[t]);
                    }
                });
            }

            for (auto& t : threads){
                t.join();
            }

            for (Network grad : thread_gradients){
                update_net(net,grad,batch_size,learning_rate);
            }

            data_count += batch_size;
        }

        auto end = chrono::system_clock::now();
        auto time = chrono::duration_cast<chrono::seconds>(end - begin);
        cout << "Time to complete: " << time.count() << " s" << endl; 


        if ((epoch + 1) % 5 == 0){
            string file = save_dir + "\\MyNeuralNet_" + to_string(epoch + 1) + "_EPOCHS.csv";
            write_net_to_csv(net,file);
        }        
    }
}

string create_now_dir(string path){
    auto now = chrono::system_clock::now();
    auto time = chrono::system_clock::to_time_t(now);
    
    stringstream ss;
    ss << put_time(localtime(&time), "%m-%d %H-%M");
    string time_f = ss.str();

    try {
        filesystem::create_directory(path + "\\" + time_f);
    } catch (const filesystem::filesystem_error& e) {
        std::cerr << "Error creating directory: " << e.what() << std::endl;
    }
    return path + "/" + time_f;
}

int main() {
    string dir = create_now_dir(filesystem::absolute(__FILE__).parent_path().string() + "\\neurals");
    
    vector<Matrix<float,input_size,1>> images;
    readImages(images);
    
    vector<uint8_t> labels_uint8;
    readLabels(labels_uint8);

    vector<int> labels;
    labels.resize(labels_uint8.size(),-1);
    for (int i = 0; i < labels_uint8.size(); i++){
        labels[i] = static_cast<int>(labels_uint8[i]);
    }

    Network net;
    net.HeDist();
    // net.fromCSV("neurals\\old\\MyNeuralNet_30_EPOCHS.csv");

    // write_net_to_csv(net,"TEST.csv");

    train(net,images,labels,dir);

    return 0;
}
