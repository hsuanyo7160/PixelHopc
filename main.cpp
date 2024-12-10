#include <stdio.h>
#include <vector>   // for vector
#include <chrono>   // for high_resolution_clock
#include <string>   // for string
#include <iostream> // for cout
#include <algorithm>
#include <numeric>
#include <map>
using namespace std;

using FeatureType = vector<vector<vector<vector<int>>>>;
using Matrix = vector<vector<float>>;
// 1. PixelHop_Unit函式

FeatureType PixelHop_8_Neighbour(FeatureType &feature, int dilate, const string &pad) {
    cout << "------------------- Start: PixelHop_8_Neighbour\n";
    cout << "       <Info>        Input feature size: " << feature.size() << "\n";
    cout << "       <Info>        dilate: " << dilate << "\n";
    cout << "       <Info>        padding: " << pad << "\n";
    auto t0 = chrono::high_resolution_clock::now();

    // Determine shape
    int batch_size = feature.size();
    int height = feature[0].size();
    int width = feature[0][0].size();
    int depth = feature[0][0][0].size();

    // Apply padding
    FeatureType padded_feature;
    if (pad == "reflect") {
        padded_feature.resize(batch_size);
        for (int b = 0; b < batch_size; ++b) {
            padded_feature[b].resize(height + 2 * dilate, vector<vector<int>>(width + 2 * dilate, vector<int>(depth)));

            for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                    for (int d = 0; d < depth; ++d) {
                        padded_feature[b][i + dilate][j + dilate][d] = feature[b][i][j][d];
                    }
                }
            }
            // Reflect padding
            for (int i = 0; i < dilate; ++i) {
                padded_feature[b][i] = padded_feature[b][2 * dilate - i - 1];
                padded_feature[b][height + dilate + i] = padded_feature[b][height + dilate - i - 1];
            }
            for (int i = 0; i < height + 2 * dilate; ++i) {
                for (int j = 0; j < dilate; ++j) {
                    padded_feature[b][i][j] = padded_feature[b][i][2 * dilate - j - 1];
                    padded_feature[b][i][width + dilate + j] = padded_feature[b][i][width + dilate - j - 1];
                }
            }
        }
    } else if (pad == "zeros") {
        padded_feature.resize(batch_size, vector<vector<vector<int>>>(height + 2 * dilate,
            vector<vector<int>>(width + 2 * dilate, vector<int>(depth, 0))));
        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                    for (int d = 0; d < depth; ++d) {
                        padded_feature[b][i + dilate][j + dilate][d] = feature[b][i][j][d];
                    }
                }
            }
        }
    } else {
        padded_feature = feature;
    }

    // Output shape
    FeatureType res;
    if (pad == "none") {
        res.resize(batch_size, vector<vector<vector<int>>>(height - 2 * dilate,
            vector<vector<int>>(width - 2 * dilate, vector<int>(9 * depth))));
    } else {
        res.resize(batch_size, vector<vector<vector<int>>>(height,
            vector<vector<int>>(width, vector<int>(9 * depth))));
    }

    // Extract 8 neighbors
    vector<int> idx = {-1, 0, 1};
    for (int b = 0; b < batch_size; ++b) {
        for (int i = dilate; i < padded_feature[b].size() - dilate; ++i) {
            for (int j = dilate; j < padded_feature[b][i].size() - dilate; ++j) {
                vector<int> tmp;
                for (int ii : idx) {
                    for (int jj : idx) {
                        int iii = i + ii * dilate;
                        int jjj = j + jj * dilate;
                        tmp.insert(tmp.end(), padded_feature[b][iii][jjj].begin(), padded_feature[b][iii][jjj].end());
                    }
                }
                res[b][i - dilate][j - dilate] = tmp;
            }
        }
    }

    cout << "       <Info>        Output feature size: " << res.size() << "\n";
    auto t1 = chrono::high_resolution_clock::now();
    cout << "------------------- End: PixelHop_8_Neighbour -> using "
         << chrono::duration<double>(t1 - t0).count() << " seconds\n";
    return res;
}

// Pixelhop_fit 函式
Matrix Pixelhop_fit(const string &weight_path, const Matrix &feature, bool useDC) {
    cout << "------------------- Start: Pixelhop_fit\n";
    cout << "       <Info>        Using weight: " << weight_path << "\n";
    auto t0 = chrono::high_resolution_clock::now();
    
    int rows = 6, cols = feature[0].size();
    //////////////////////////建立權重與偏移量//////////////////////////   需要處理外來檔案
    Matrix weight(rows, vector<float>(cols, 0.5)); // 模擬隨機權重
    vector<float> bias(rows, 0.1); // 模擬偏移量
    ///////////////////////////////////////////////////////////////////
    Matrix feature_w_bias = feature;
    for (size_t i = 0; i < feature_w_bias.size(); ++i) {
        for (size_t j = 0; j < feature_w_bias[i].size(); ++j) {
            feature_w_bias[i][j] += 1 / sqrt(cols) * bias[j % rows];
        }
    }

    Matrix transformed_feature(feature_w_bias.size(), vector<float>(rows, 0.0));
    for (size_t i = 0; i < feature_w_bias.size(); ++i) {
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                transformed_feature[i][r] += feature_w_bias[i][c] * weight[r][c];
            }
        }
    }

    if (useDC) {
        for (size_t i = 0; i < transformed_feature.size(); ++i) {
            transformed_feature[i][0] -= bias[0];
        }
    }

    cout << "       <Info>        Transformed feature shape: (" 
         << transformed_feature.size() << ", " << transformed_feature[0].size() << ")\n";
    auto t1 = chrono::high_resolution_clock::now();
    cout << "------------------- End: Pixelhop_fit -> using "
         << chrono::duration<double>(t1 - t0).count() << " seconds\n";
    return transformed_feature;
}

// PixelHop_Unit函式        tmp.pkl需要換掉
Matrix PixelHop_Unit(FeatureType &feature, int dilate = 1, int num_AC_kernels = 6, 
                            const string &pad = "reflect", const string &weight_name = "tmp.pkl", 
                            bool getK = false, bool useDC = false) {
    cout << "Feature shape: " << feature.size() << ", " << feature[0].size() << ", "
         << feature[0][0].size() << ", " << feature[0][0][0].size() << "\n";

    auto feature_transformed = PixelHop_8_Neighbour(feature, dilate, pad);

    int batch_size = feature_transformed.size();
    int height = feature_transformed[0].size();
    int width = feature_transformed[0][0].size();
    int depth = feature_transformed[0][0][0].size();
    Matrix feature_matrix(batch_size * height * width, vector<float>(depth, 0.0));

    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                for (int d = 0; d < depth; ++d) {
                    feature_matrix[b * height * width + i * width + j][d] = feature_transformed[b][i][j][d];
                }
            }
        }
    }

    cout << "Feature shape: " << feature_matrix.size() << ", " << feature_matrix[0].size() << "\n";
    return Pixelhop_fit("../weight/" + weight_name, feature_matrix, useDC);
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
FeatureType manual_block_reduce(const FeatureType& input_array, const pair<int, int>& block_size, 
                                float (*func)(const Matrix&)) {
    // 獲取輸入數據的形狀
    int batch_size = input_array.size();
    int height = input_array[0].size();
    int width = input_array[0][0].size();
    int channels = input_array[0][0][0].size();

    // 計算新的尺寸
    int new_height = height / block_size.first;
    int new_width = width / block_size.second;

    // 初始化結果矩陣
    FeatureType output_array;
    // 初始化第 1 層（batch_size）
    output_array.resize(batch_size);
    for (int b = 0; b < batch_size; ++b) {
        // 初始化第 2 層（new_height）
        output_array[b].resize(new_height);
        for (int i = 0; i < new_height; ++i) {
            // 初始化第 3 層（new_width）
            output_array[b][i].resize(new_width);
            for (int j = 0; j < new_width; ++j) {
                // 初始化第 4 層（channels），並設為 0.0f
                output_array[b][i][j].resize(channels, 0.0f);
            }
        }
    }

    // 使用切片操作來進行區塊處理
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int i = 0; i < new_height; ++i) {
                for (int j = 0; j < new_width; ++j) {
                    // 提取區塊並存入臨時矩陣
                    Matrix block(block_size.first, vector<float>(block_size.second, 0.0));
                    for (int bi = 0; bi < block_size.first; ++bi) {
                        for (int bj = 0; bj < block_size.second; ++bj) {
                            int orig_i = i * block_size.first + bi;
                            int orig_j = j * block_size.second + bj;
                            block[bi][bj] = input_array[b][orig_i][orig_j][c];
                        }
                    }
                    // 計算函數值（例如平均值）並儲存
                    output_array[b][i][j][c] = func(block);
                }
            }
        }
    }

    return output_array;
}

// 3.
// Normalization函式
Matrix Norm(const Matrix& feature) {
    Matrix normalized(feature.size(), vector<float>(feature[0].size(), 0.0f));
    for (size_t i = 0; i < feature.size(); ++i) {
        float min_val = *min_element(feature[i].begin(), feature[i].end());
        float sum = 0.0f;
        for (float val : feature[i]) 
            sum += (val - min_val);

        for (size_t j = 0; j < feature[i].size(); ++j) {
            normalized[i][j] = (feature[i][j] - min_val) / sum;
        }
    }
    return normalized;
}

// ReLU 函數
Matrix Relu(const Matrix& input) {
    Matrix result = input;
    for (auto& row : result) {
        for (auto& val : row) {
            val = max(0.0f, val);
        }
    }
    return result;
}

// KMeans 類的簡易實現
class KMeans {
public:
    KMeans(int clusters, int batch_size) : clusters_(clusters), batch_size_(batch_size) {}
    void fit(const Matrix& data) {
        // 假設這裡是一個 KMeans 聚類的實現
        // 填充 centroids_ 和 labels_
    }
    const Matrix& get_centroids() const { return centroids_; }
    const vector<int>& get_labels() const { return labels_; }

private:
    int clusters_;
    int batch_size_;
    Matrix centroids_;
    vector<int> labels_;
};

// 計算目標函數
void compute_target(const Matrix& feature, const vector<int>& train_labels, int num_clusters, const vector<int>& class_list, 
                    vector<int>& labels, vector<int>& clus_labels, Matrix& centroids) {
    int use_classes = class_list.size();
    int num_clusters_sub = num_clusters / use_classes;
    int batch_size = 1000;

    labels.resize(feature.size());
    clus_labels.resize(num_clusters);
    centroids.resize(num_clusters, vector<float>(feature[0].size(), 0.0f));

    for (int i = 0; i < use_classes; ++i) {
        int class_id = class_list[i];
        // 選擇屬於該類的樣本
        vector<int> indices;
        for (size_t j = 0; j < train_labels.size(); ++j) {
            if (train_labels[j] == class_id) {
                indices.push_back(j);
            }
        }

        // 提取屬於該類的樣本
        Matrix feature_train(indices.size(), vector<float>(feature[0].size(), 0.0f));
        for (size_t j = 0; j < indices.size(); ++j) {
            feature_train[j] = feature[indices[j]];
        }

        // 聚類
        KMeans kmeans(num_clusters_sub, batch_size);
        kmeans.fit(feature_train);

        // 保存結果
        for (size_t j = 0; j < indices.size(); ++j) {
            labels[indices[j]] = kmeans.get_labels()[j] + i * num_clusters_sub;
        }
        copy(kmeans.get_labels().begin(), kmeans.get_labels().end(), clus_labels.begin() + i * num_clusters_sub);
        copy(kmeans.get_centroids().begin(), kmeans.get_centroids().end(), centroids.begin() + i * num_clusters_sub);

        cout << "FINISH KMEANS " << i << endl;
    }
}

// 線性回歸訓練
void llsr_train(const Matrix& feature_train, const vector<int>& labels_train, Matrix& weights, Matrix& bias, int alpha) {
    // 假設這裡是一個線性回歸的簡易實現
    // 根據 feature_train 和 labels_train 訓練模型，填充 weights 和 bias
}

// 線性回歸測試
Matrix llsr_test(const Matrix& feature_test, const Matrix& weights, const Matrix& bias) {
    Matrix result = feature_test;
    for (size_t i = 0; i < result.size(); ++i) {
        for (size_t j = 0; j < weights[0].size(); ++j) {
            result[i][j] = inner_product(result[i].begin(), result[i].end(), weights[j].begin(), 0.0f) + bias[0][j];
        }
    }
    return result;
}

// LAG 單元
Matrix LAG_Unit_manual(const Matrix& feature, const vector<int>& train_labels, const vector<int>& class_list, map<string, Matrix>& SAVE,
                       int num_clusters = 50, int alpha = 5, bool Train = true) {
    if (Train) {
        cout << "--------Train LAG Unit--------" << endl;
        cout << "feature_train shape: (" << feature.size() << ", " << feature[0].size() << ")" << endl;

        vector<int> labels_train, clus_labels;
        Matrix centroids;

        // 計算聚類目標
        compute_target(feature, train_labels, num_clusters, class_list, labels_train, clus_labels, centroids);

        // 標準化
        Matrix scaler = Norm(feature);

        // 訓練線性回歸
        Matrix weights, bias;
        llsr_train(scaler, labels_train, weights, bias, alpha);

        // 保存模型
        SAVE["clus_labels"] = centroids;
        SAVE["LLSR weight"] = weights;
        SAVE["LLSR bias"] = bias;

        cout << "weight shape: (" << weights.size() << ", " << weights[0].size() << ")" << endl;
        cout << "bias shape: (" << bias.size() << ", " << bias[0].size() << ")" << endl;

        // 測試
        Matrix feature_transformed = llsr_test(scaler, weights, bias);
        return feature_transformed;
    } else {
        cout << "--------Testing--------" << endl;
        Matrix weights = SAVE["LLSR weight"];
        Matrix bias = SAVE["LLSR bias"];
        Matrix scaler = SAVE["scaler"];

        Matrix feature_reduced = llsr_test(scaler, weights, bias);
        return feature_reduced;
    }
}




int main() {
    
}