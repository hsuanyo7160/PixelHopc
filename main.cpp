#include <stdio.h>
#include <vector>   // for vector
#include <chrono>   // for high_resolution_clock
#include <string>   // for string
#include <iostream> // for cout
#include <algorithm>
#include <numeric>
#include <map>
#include <functional>
using namespace std;

using FeatureType = vector<vector<vector<vector<float>>>>;
using Matrix = vector<vector<float>>;
// 1. PixelHop_Unit函式

FeatureType PixelHop_8_Neighbour(const FeatureType &feature, int dilate, const string &pad) {
    cout << "------------------- Start: PixelHop_8_Neighbour" << endl;
    cout << "       <Info>        Input feature shape: " << feature.size() << endl;
    cout << "       <Info>        dilate: " << dilate << endl;
    cout << "       <Info>        padding: " << pad << endl;

    // 獲取維度
    int batch_size = feature.size();
    int height = feature[0].size();
    int width = feature[0][0].size();
    int depth = feature[0][0][0].size();

    // 填充處理
    FeatureType padded_feature;
    if (pad == "reflect") {
        padded_feature.resize(batch_size, vector<vector<vector<float>>>(
            height + 2 * dilate, vector<vector<float>>(
                width + 2 * dilate, vector<float>(depth))));

        for (int b = 0; b < batch_size; b++) {
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    for (int d = 0; d < depth; d++) {
                        // 填充反射數據
                        padded_feature[b][i + dilate][j + dilate][d] = feature[b][i][j][d];
                    }
                }
            }
        }
    } else if (pad == "zeros") {
        padded_feature.resize(batch_size, vector<vector<vector<float>>>(
            height + 2 * dilate, vector<vector<float>>(
                width + 2 * dilate, vector<float>(depth, 0))));
    } else { // pad == "none"
        padded_feature = feature;
    }

    // 輸出結果初始化
    FeatureType res(batch_size, vector<vector<vector<float>>>(
        height, vector<vector<float>>(
            width, vector<float>(9 * depth, 0))));

    // 提取 8 鄰域
    vector<int> idx = {-1, 0, 1};
    for (int b = 0; b < batch_size; b++) {
        for (int i = dilate; i < height + dilate; i++) {
            for (int j = dilate; j < width + dilate; j++) {
                vector<float> tmp;
                for (int ii : idx) {
                    for (int jj : idx) {
                        int iii = i + ii * dilate;
                        int jjj = j + jj * dilate;
                        tmp.insert(tmp.end(),
                                   padded_feature[b][iii][jjj].begin(),
                                   padded_feature[b][iii][jjj].end());
                    }
                }
                res[b][i - dilate][j - dilate] = tmp;
            }
        }
    }

    cout << "       <Info>        Output feature shape: " << res.size() << endl;
    cout << "------------------- End: PixelHop_8_Neighbour---------------- "<< endl;

    return res;
}

// 適配函數
FeatureType Pixelhop_fit(const string &weight_path, const FeatureType &feature, bool useDC) {
    cout << "------------------- Start: Pixelhop_fit ------------------" << endl;
    cout << "       <Info>        Using weight: " << weight_path << endl;

    // ifstream weight_file(weight_path, ios::binary);
    // if (!weight_file) {
    //     cerr << "Error: Unable to open weight file." << endl;
    //     exit(1);
    // }

    // 假設讀取 PCA 參數（此處需具體實現）
    vector<float> weight; // 模擬的權重
    vector<float> bias;   // 模擬的偏差

    // 模擬權重運算
    FeatureType transformed_feature = feature; // 假設此處運算完成
    if (useDC) {
        // 使用 DC 處理
    }

    cout << "------------------- End: Pixelhop_fit " << endl;

    return transformed_feature;
}

// PixelHop 單元實現
FeatureType PixelHop_Unit_manual(const FeatureType &feature, int dilate = 1, int num_AC_kernels = 6,
                              const string &pad = "reflect", const string &weight_name = "tmp.pkl",
                              bool getK = false, bool useDC = false) {
    cout << "Feature shape: " << feature.size() << ", " << feature[0].size() << ", "
         << feature[0][0].size() << ", " << feature[0][0][0].size() << endl;

    auto processed_feature = PixelHop_8_Neighbour(feature, dilate, pad);

    // PCA 或其他權重處理
    auto transformed_feature = Pixelhop_fit(weight_name, processed_feature, useDC);
    
    // GetK (always false)
    if(getK) {
        // 取得 Kernels
    }
    return transformed_feature;
}
// 2. 
// 用於計算區塊平均值的函數 for func parameter
float compute_average(const Matrix& block) {
    float sum = 0.0;
    int total_elements = block.size() * block[0].size();
    for (const auto& row : block) {
        for (const auto& val : row) {
            sum += val;
        }
    }
    return sum / total_elements;
}
// 手動區塊縮減函數
float mean(const vector<float>& block) {
    float sum = accumulate(block.begin(), block.end(), 0.0f);
    return sum / block.size();
}

FeatureType manual_block_reduce(const FeatureType &input_array, vector<int> &block_size, function<float(const vector<float>&)> func){
    // 獲取輸入數據的形狀
    int batch_size = input_array.size();
    int height = input_array[0].size();
    int width = input_array[0][0].size();
    int channels = input_array[0][0][0].size();

    // 計算新的尺寸
    int new_height = height / block_size[0];
    int new_width = width / block_size[1];

    // 初始化結果矩陣
    FeatureType output_array(batch_size, vector<vector<vector<float>>>(new_height, vector<vector<float>>(new_width, vector<float>(channels, 0.0f))));

    // 使用切片操作進行區塊處理
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int i = 0; i < new_height; ++i) {
                for (int j = 0; j < new_width; ++j) {
                    // 提取區塊數據
                    vector<float> block;
                    for (int bi = 0; bi < block_size[0]; ++bi) {
                        for (int bj = 0; bj < block_size[1]; ++bj) {
                            block.push_back(input_array[b][i * block_size[0] + bi][j * block_size[1] + bj][c]);
                        }
                    }
                    // 使用提供的函數對區塊進行操作
                    output_array[b][i][j][c] = func(block);
                }
            }
        }
    }

    return output_array;
}

// 3.
// Lag_Unit 函數
class StandardScaler {
public:
    void fit(const Matrix& data) {
    int rows = data.size(), cols = data[0].size();
    mean.resize(cols, 0.0f);
    std_dev.resize(cols, 0.0f);

    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            mean[j] += data[i][j];
        }
        mean[j] /= rows;

        for (int i = 0; i < rows; ++i) {
            std_dev[j] += pow(data[i][j] - mean[j], 2);
        }
        std_dev[j] = sqrt(std_dev[j] / rows);
        if (std_dev[j] < 1e-8) {  // 防止除以零
            std_dev[j] = 1.0;
        }
    }
}

    Matrix transform(const Matrix& data) const {
        int rows = data.size(), cols = data[0].size();
        Matrix result = data;

        for (int j = 0; j < cols; ++j) {
            for (int i = 0; i < rows; ++i) {
                result[i][j] = (data[i][j] - mean[j]) / std_dev[j];
            }
        }

        return result;
    }

private:
    vector<float> mean;
    vector<float> std_dev;
};

// ReLU 函數
Matrix Relu(const Matrix& data) {
    Matrix result = data;
    for (auto& row : result) {
        for (auto& val : row) {
            val = max(0.0f, val);
        }
    }
    return result;
}

// 最小二乘法解算器
void least_squares(const Matrix& X, const Matrix& y, Matrix& weights, Matrix& bias) {
    int rows = X.size(), cols = X[0].size();
    int target_dim = y[0].size();

    // 初始化
    weights.resize(cols, vector<float>(target_dim, 0.0f));
    bias.resize(1, vector<float>(target_dim, 0.0f));

    for (int t = 0; t < target_dim; ++t) {
        for (int j = 0; j < cols; ++j) {
            for (int i = 0; i < rows; ++i) {
                weights[j][t] += X[i][j] * y[i][t];
            }
            weights[j][t] /= rows;
        }

        for (int i = 0; i < rows; ++i) {
            bias[0][t] += y[i][t];
        }
        bias[0][t] /= rows;
    }
}

// KMeans 聚類
class KMeans {
public:
    KMeans(int n_clusters, int max_iter = 100) : n_clusters(n_clusters), max_iter(max_iter) {}

    void fit(const Matrix& data) {
        int rows = data.size(), cols = data[0].size();
        centroids.resize(n_clusters, vector<float>(cols, 0.0f));

        for (int i = 0; i < n_clusters; ++i) {
            centroids[i] = data[rand() % rows];
        }

        for (int iter = 0; iter < max_iter; ++iter) {
            vector<int> new_labels(rows, 0);
            Matrix new_centroids(n_clusters, vector<float>(cols, 0.0f));
            vector<int> counts(n_clusters, 0);

            // 分配每個點到最近的中心點
            for (int i = 0; i < rows; ++i) {
                float min_dist = numeric_limits<float>::max();
                for (int k = 0; k < n_clusters; ++k) {
                    float dist = 0.0f;
                    for (int j = 0; j < cols; ++j) {
                        dist += pow(data[i][j] - centroids[k][j], 2);
                    }
                    if (dist < min_dist) {
                        min_dist = dist;
                        new_labels[i] = k;
                    }
                }

                counts[new_labels[i]]++;
                for (int j = 0; j < cols; ++j) {
                    new_centroids[new_labels[i]][j] += data[i][j];
                }
            }

            // 更新中心點
            for (int k = 0; k < n_clusters; ++k) {
                for (int j = 0; j < cols; ++j) {
                    centroids[k][j] = counts[k] ? new_centroids[k][j] / counts[k] : centroids[k][j];
                }
            }

            if (new_labels == labels) break; // 收斂
            labels = new_labels;
        }
    }

    const vector<int>& get_labels() const { return labels; }
    const Matrix& get_centroids() const { return centroids; }

private:
    int n_clusters;
    int max_iter;
    vector<int> labels;
    Matrix centroids;
};

// LAG_Unit_manual 函數
Matrix LAG_Unit_manual(const Matrix& feature, const vector<int>& train_labels, const vector<int>& class_list, 
                       map<string, Matrix>& SAVE, int num_clusters = 50, int alpha = 5, bool Train = true) {
    if (Train) {
        cout << "--------Train LAG Unit--------" << endl;
        int use_classes = class_list.size();
        int num_clusters_sub = num_clusters / use_classes;

        vector<int> labels_train;
        Matrix clus_labels, centroids;

        // KMeans 聚類
        KMeans kmeans(num_clusters);
        kmeans.fit(feature);
        labels_train = kmeans.get_labels();
        centroids = kmeans.get_centroids();

        // 標準化
        StandardScaler scaler;
        scaler.fit(feature);
        Matrix feature_norm = scaler.transform(feature);

        // 線性回歸
        Matrix weights, bias;
        least_squares(feature_norm, Matrix(train_labels.begin(), train_labels.end()), weights, bias);

        // 保存模型
        SAVE["clus_labels"] = centroids;
        SAVE["LLSR weight"] = weights;
        SAVE["LLSR bias"] = bias;

        cout << "Training completed!" << endl;
        return feature_norm;
    } else {
        cout << "--------Testing--------" << endl;
        Matrix weights = SAVE["LLSR weight"];
        Matrix bias = SAVE["LLSR bias"];
        StandardScaler scaler;
        Matrix feature_test = scaler.transform(feature);

        Matrix result; // 測試結果
        return result;
    }
}

//4. 手刻SVM
class MyLinearSVM {
public:
    MyLinearSVM(float learning_rate = 0.01, float lambda_reg = 0.01, int num_iters = 1000)
        : learning_rate(learning_rate), lambda_reg(lambda_reg), num_iters(num_iters) {
        bias = 0.0f;
    }

    void fit(const Matrix& X, const vector<int>& y) {
        size_t num_samples = X.size();
        size_t num_features = X[0].size();
        weights = vector<float>(num_features, 0.0f);

        vector<int> adjusted_y(y);
        for (auto& label : adjusted_y) {
            label = (label <= 0) ? -1 : 1;
        }

        for (int iter = 0; iter < num_iters; ++iter) {
            vector<float> dw(num_features, 0.0f);
            float db = 0.0f;

            for (size_t i = 0; i < num_samples; ++i) {
                float score = inner_product(X[i].begin(), X[i].end(), weights.begin(), bias);
                if (adjusted_y[i] * score < 1) {
                    for (size_t j = 0; j < num_features; ++j) {
                        dw[j] -= adjusted_y[i] * X[i][j];
                    }
                    db -= adjusted_y[i];
                }
            }

            for (size_t j = 0; j < num_features; ++j) {
                dw[j] = dw[j] / num_samples + lambda_reg * weights[j];
            }
            db /= num_samples;

            for (size_t j = 0; j < num_features; ++j) {
                weights[j] -= learning_rate * dw[j];
            }
            bias -= learning_rate * db;
        }
    }

    vector<int> predict(const Matrix& X) {
        vector<int> predictions(X.size());
        for (size_t i = 0; i < X.size(); ++i) {
            float score = inner_product(X[i].begin(), X[i].end(), weights.begin(), bias);
            predictions[i] = (score >= 0) ? 1 : -1;
        }
        return predictions;
    }

    vector<float> weights;
    float bias;

private:
    float learning_rate;
    float lambda_reg;
    int num_iters;
};

class MySVC {
public:
    MySVC(float learning_rate = 0.01, float lambda_reg = 0.01, int num_iters = 1000)
        : learning_rate(learning_rate), lambda_reg(lambda_reg), num_iters(num_iters) {}

    void fit(const Matrix& X, const vector<int>& y) {
        map<int, MyLinearSVM> models;
        vector<int> classes;
        for (int label : y) {
            if (find(classes.begin(), classes.end(), label) == classes.end()) {
                classes.push_back(label);
            }
        }

        for (int c : classes) {
            vector<int> binary_labels(y.size());
            for (size_t i = 0; i < y.size(); ++i) {
                binary_labels[i] = (y[i] == c) ? 1 : -1;
            }

            MyLinearSVM svm(learning_rate, lambda_reg, num_iters);
            svm.fit(X, binary_labels);
            models[c] = svm;
        }

        this->models = models;
    }

    vector<int> predict(const Matrix& X) {
        Matrix scores(X.size(), vector<float>(models.size()));
        vector<int> classes;
        for (const auto& pair : models) {
            classes.push_back(pair.first);
        }

        for (size_t i = 0; i < X.size(); ++i) {
            size_t class_idx = 0;
            for (const auto& pair : models) {
                scores[i][class_idx++] = inner_product(X[i].begin(), X[i].end(), pair.second.weights.begin(), pair.second.bias);
            }
        }

        vector<int> predictions(X.size());
        for (size_t i = 0; i < scores.size(); ++i) {
            size_t max_idx = distance(scores[i].begin(), max_element(scores[i].begin(), scores[i].end()));
            predictions[i] = classes[max_idx];
        }

        return predictions;
    }

    float accuracy(const Matrix& X, const vector<int>& y) {
        vector<int> predictions = predict(X);
        int correct = 0;
        for (size_t i = 0; i < y.size(); ++i) {
            if (predictions[i] == y[i]) {
                ++correct;
            }
        }
        return static_cast<float>(correct) / y.size();
    }

private:
    map<int, MyLinearSVM> models;
    float learning_rate;
    float lambda_reg;
    int num_iters;
};


// Main function
vector<float> flatten(const vector<vector<vector<float>>>& feature) {
    vector<float> result;
    for (const auto& matrix : feature) {
        for (const auto& row : matrix) {
            result.insert(result.end(), row.begin(), row.end());
        }
    }
    return result;
}

vector<float> computeMean(const Matrix& data) {
    vector<float> mean(data[0].size(), 0.0f);
    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            mean[i] += row[i];
        }
    }
    for (auto& val : mean) {
        val /= data.size();
    }
    return mean;
}

Matrix standardize(const Matrix& data, const vector<float>& mean, const vector<float>& std) {
    Matrix result = data;
    for (auto& row : result) {
        for (size_t i = 0; i < row.size(); ++i) {
            row[i] = (row[i] - mean[i]) / std[i];
        }
    }
    return result;
}

vector<float> computeStd(const Matrix& data, const vector<float>& mean) {
    vector<float> std(data[0].size(), 0.0f);
    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            std[i] += pow(row[i] - mean[i], 2);
        }
    }
    for (auto& val : std) {
        val = sqrt(val / data.size());
    }
    return std;
}
int main() {
    // 假設載入 MNIST 資料的函式為 import_data_mnist
    FeatureType train_images, test_images;
    vector<int> train_labels, test_labels;
    vector<int> class_list;
    size_t N_train = 1000;
    size_t N_test = 500;

    // 載入 MNIST 數據
    //import_data_mnist("0-9", train_images, train_labels, test_images, test_labels, class_list);

    // 限制訓練和測試數據大小
    train_images.resize(N_train);
    train_labels.resize(N_train);
    test_images.resize(N_test);
    test_labels.resize(N_test);

    // 生成訓練特徵
    auto train_feature = PixelHop_Unit_manual(train_images, 1, 5, "reflect", "pixelhop1_mnist.pkl", true);
    train_feature = manual_block_reduce(train_feature, {1, 4, 4, 1}, [](const auto& block) {
        return mean(block); // 計算區塊的平均值
    });

    // 將特徵展平為 2D
    Matrix train_feature_flat(N_train, vector<float>(train_feature[0][0][0].size()));
    for (size_t i = 0; i < N_train; ++i) {
        train_feature_flat[i] = flatten(train_feature[i]);
    }

    // 使用 LAG_Unit_manual 降維
    map<string, Matrix> SAVE;
    auto train_feature_reduce = LAG_Unit_manual(train_feature_flat, train_labels, class_list, SAVE, 50, 5, true);

    // 生成測試特徵
    auto test_feature = PixelHop_Unit_manual(test_images, 1, 5, "reflect", "pixelhop1_mnist.pkl", false);
    test_feature = manual_block_reduce(test_feature, {1, 4, 4, 1}, [](const auto& block) {
        return mean(block); // 計算區塊的平均值
    });

    // 將特徵展平為 2D
    Matrix test_feature_flat(N_test, vector<float>(test_feature[0][0][0].size()));
    for (size_t i = 0; i < N_test; ++i) {
        test_feature_flat[i] = flatten(test_feature[i]);
    }

    // 使用 LAG_Unit_manual 降維
    auto test_feature_reduce = LAG_Unit_manual(test_feature_flat, {}, class_list, SAVE, 50, 5, false);

    // 計算訓練特徵的平均值和標準差
    auto mean = computeMean(train_feature_reduce);
    auto std = computeStd(train_feature_reduce, mean);

    // 標準化訓練和測試特徵
    auto standardized_train = standardize(train_feature_reduce, mean, std);
    auto standardized_test = standardize(test_feature_reduce, mean, std);

    // 使用支持向量機 (SVC) 進行訓練
    MySVC clf_manual(0.01, 0.01, 1000);
    clf_manual.fit(standardized_train, train_labels);

    // 計算訓練和測試的準確率
    float train_acc = clf_manual.accuracy(standardized_train, train_labels);
    float test_acc = clf_manual.accuracy(standardized_test, test_labels);

    // 輸出結果
    cout << "***** Train ACC: " << train_acc << endl;
    cout << "***** Test ACC: " << test_acc << endl;

    return 0;
}









