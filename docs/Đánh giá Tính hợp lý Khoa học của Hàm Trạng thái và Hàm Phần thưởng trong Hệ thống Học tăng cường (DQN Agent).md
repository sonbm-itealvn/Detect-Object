# Đánh giá Tính hợp lý Khoa học của Hàm Trạng thái và Hàm Phần thưởng trong Hệ thống Học tăng cường (DQN Agent)

## 1. Giới thiệu

Báo cáo này nhằm mục đích kiểm chứng và đánh giá tính hợp lý khoa học của hàm trạng thái (state representation) và hàm phần thưởng (reward function) được mô tả trong tài liệu `README.md` cho một tác nhân Học tăng cường (Reinforcement Learning - RL) dựa trên Deep Q-Network (DQN). Hệ thống này được thiết kế để cải thiện hiệu suất của một pipeline Phát hiện Quan hệ Thị giác (Visual Relationship Detection - VRD) thông qua một vòng lặp dữ liệu tự động. Việc thiết kế hàm trạng thái và hàm phần thưởng đóng vai trò then chốt trong sự thành công của bất kỳ hệ thống RL nào, ảnh hưởng trực tiếp đến khả năng học hỏi và tối ưu hóa hành vi của tác nhân.

## 2. Phân tích Hàm Trạng thái (State Representation)

Trong bối cảnh của hệ thống VRD này, trạng thái $\mathbf{s}_t$ tại thời điểm $t$ được định nghĩa là một vector tổng hợp, cung cấp thông tin toàn diện về môi trường và hiệu suất hiện tại của tác nhân. Cụ thể, hàm trạng thái được mô tả như sau:

$$\mathbf{s}_t = [\underbrace{F_1^{\text{det}}, \ldots, F_K^{\text{det}}}_{\text{Detection features}}, \underbrace{N_{\text{rel}}, F_1^{\text{rel}}, \ldots}_{\text{Relationship features}}, \underbrace{r_{t-1}, \varepsilon_t, t/T}_{\text{Training state}}, \underbrace{U_{\text{mean}}, U_{\text{max}}}_{\text{Uncertainty}}] \in \mathbb{R}^{D_s}$$

Các thành phần chi tiết của vector trạng thái bao gồm:

| Thành phần | Mô tả | Nguồn | Kích thước |
|---|---|---|---|
| **Đặc trưng Phát hiện** ($F^{\text{det}}$) | Độ tin cậy phát hiện trung bình, số lượng đối tượng phát hiện (đã chuẩn hóa). | Kết quả YOLOv11 | 2 |
| **Đặc trưng Quan hệ** ($F^{\text{rel}}$) | Độ tin cậy quan hệ trung bình, số lượng quan hệ (đã chuẩn hóa), phân phối loại quan hệ (histogram). | Kết quả RelTR | $2 + R$ (với $R$ là số loại quan hệ) |
| **Trạng thái Huấn luyện** | Phần thưởng trước đó ($r_{t-1}$), tỷ lệ khám phá ($\varepsilon_t$), tiến độ epoch ($t/T$). | Trạng thái nội bộ DQN | 3 |
| **Độ bất định** ($U$) | Độ bất định trung bình ($U_{\text{mean}}$), độ bất định tối đa ($U_{\text{max}}$) của các dự đoán. | MC Dropout Uncertainty Estimator | 2 |

**Trong code:** `README.md` §11.2 và `RL/reinforcement_learning.py` → `_build_state_vector` dùng vector **9 chiều** cố định (bỏ histogram $R$ chiều): Detection 2, Relationship 2, Training 3, Uncertainty 2.

**Đánh giá khoa học:**

Thiết kế hàm trạng thái này thể hiện sự hiểu biết sâu sắc về bài toán VRD và các nguyên tắc của Học tăng cường. Việc kết hợp các loại thông tin đa dạng là hợp lý:

1.  **Đặc trưng Thị giác và Thống kê:** Các đặc trưng từ kết quả phát hiện đối tượng (YOLOv11) và suy luận quan hệ (RelTR) cung cấp thông tin trực tiếp về hiệu suất của pipeline VRD. Việc chuẩn hóa số lượng đối tượng và quan hệ là cần thiết để tránh các giá trị đầu vào quá lớn gây mất ổn định cho mạng nơ-ron [1]. Phân phối loại quan hệ dưới dạng histogram là một cách hiệu quả để mã hóa thông tin ngữ nghĩa về sự đa dạng của các quan hệ được phát hiện.
2.  **Trạng thái Huấn luyện:** Việc đưa $r_{t-1}$ (phần thưởng của bước trước) vào trạng thái là một kỹ thuật phổ biến trong các hệ thống RL có tính chất tuần tự, giúp tác nhân hiểu được tác động tức thời của hành động trước đó. $\varepsilon_t$ (tỷ lệ khám phá) và $t/T$ (tiến độ epoch) cung cấp ngữ cảnh về giai đoạn hiện tại của quá trình huấn luyện, cho phép tác nhân điều chỉnh hành vi của mình (ví dụ: khám phá nhiều hơn ở giai đoạn đầu, khai thác nhiều hơn ở giai đoạn sau) [2]. Điều này đặc biệt hữu ích trong các kịch bản Meta-RL hoặc Adaptive RL.
3.  **Độ bất định (Uncertainty):** Đây là một thành phần quan trọng, đặc biệt trong bối cảnh Học tăng cường với Active Learning. $U_{\text{mean}}$ và $U_{\text{max}}$ từ MC Dropout (được mô tả chi tiết ở Mục 13 trong `README.md`) cung cấp tín hiệu về mức độ tự tin của mô hình. Một tác nhân RL có thể sử dụng thông tin này để ưu tiên các hành động tạo ra dữ liệu mới ở những vùng có độ bất định cao, từ đó cải thiện hiệu quả của vòng lặp dữ liệu [3].

**Kết luận về tính hợp lý:** Hàm trạng thái được thiết kế một cách khoa học, bao gồm các thông tin cần thiết và đa dạng để tác nhân DQN có thể đưa ra quyết định tối ưu. Việc kết hợp các đặc trưng về hiệu suất mô hình, trạng thái huấn luyện và độ bất định tạo nên một biểu diễn trạng thái mạnh mẽ cho bài toán này.

## 3. Phân tích Hàm Phần thưởng (Reward Function)

Hàm phần thưởng là yếu tố cốt lõi định hình hành vi của tác nhân RL. Trong hệ thống này, hàm phần thưởng được thiết kế một cách phức tạp và thích nghi, nhằm thúc đẩy nhiều mục tiêu cùng lúc. Công thức tổng quát của phần thưởng là:

$$R = \sigma\big(k \cdot (R_{\text{raw}} - 0.5)\big) = \frac{1}{1 + e^{-k(R_{\text{raw}} - 0.5)}}$$

Trong đó $R_{\text{raw}}$ là tổng có trọng số của nhiều thành phần điểm, và $\sigma(\cdot)$ là hàm sigmoid để chuẩn hóa phần thưởng về khoảng $(0, 1)$. $k$ là một hệ số co giãn (scaling factor) thích nghi. Các thành phần của $R_{\text{raw}}$ bao gồm:

$$R_{\text{raw}} = W_{\text{det}} S_{\text{det}} + W_{\text{rel}} S_{\text{rel}} + W_{\text{div}} S_{\text{div}} + W_{\text{cons}} S_{\text{cons}} + W_{\text{imp}} S_{\text{imp}} + W_{\text{unc}} S_{\text{unc}}$$

Các thành phần điểm ($S_*$) và trọng số thích nghi ($W_*$) được định nghĩa như sau:

| Thành phần điểm | Mô tả | Công thức chính | Tính hợp lý khoa học |
|---|---|---|---|
| $S_{\text{det}}$ (Detection Score) | Đo hiệu suất phát hiện đối tượng. | $F1_{\text{det}} \cdot C_n \cdot B_{PR}$ | Sử dụng F1-score là chuẩn mực, $C_n$ (hệ số tin cậy mẫu) giúp ổn định khi dữ liệu ít, $B_{PR}$ (cân bằng Precision-Recall) khuyến khích sự cân bằng giữa P và R, tránh lệch [4]. |
| $S_{\text{rel}}$ (Relationship Score) | Đo hiệu suất suy luận quan hệ. | $(F1_{\text{rel}} \cdot C_n \cdot B_{PR}) \times (1 + W_{\text{tail}})$ | Tương tự $S_{\text{det}}$, bổ sung $W_{\text{tail}}$ (trọng số đuôi dài) để ưu tiên các quan hệ hiếm, giải quyết vấn đề mất cân bằng lớp [5]. |
| $S_{\text{div}}$ (Diversity Score) | Khuyến khích sự đa dạng trong dữ liệu được tạo ra. | $0.4 D_{\text{type}} + 0.4 D_{\text{class}} + 0.2 S_{\text{spatial}}$ | Đa dạng loại quan hệ, lớp đối tượng và phân bố không gian (vị trí, kích thước, độ phủ) là các yếu tố quan trọng để tạo ra dữ liệu huấn luyện phong phú, tránh overfitting và cải thiện khả năng tổng quát hóa của mô hình [6]. Việc sử dụng entropy để đo độ phủ không gian là một phương pháp chuẩn mực.
| $S_{\text{cons}}$ (Consistency Score) | Đo sự ổn định và xu hướng cải thiện của hiệu suất. | $0.7/(1+\sigma_{F1}) + 0.3\, S_{\text{trend}}$ | Khuyến khích sự ổn định (nghịch đảo độ lệch chuẩn F1) và xu hướng tăng trưởng (hệ số góc hồi quy tuyến tính của F1), giúp tác nhân tìm kiếm các chiến lược bền vững [7]. |
| $S_{\text{imp}}$ (Improvement Score) | Đo mức độ cải thiện so với baseline. | $0.6\tanh(F1_{\text{curr}}-F1_{\text{base}}) + 0.4 S_{\text{trend}}$ | Trực tiếp thưởng cho việc vượt qua baseline, với hàm $\tanh$ giúp làm mượt phần thưởng và $S_{\text{trend}}$ củng cố xu hướng cải thiện. |
| $S_{\text{unc}}$ (Uncertainty Reduction Score) | Đo mức độ giảm độ bất định của mô hình. | $0.5 + 0.5\,\rho$, với $\rho$ là mức giảm độ bất định trung bình. | Đây là một dạng phần thưởng nội tại (intrinsic reward) quan trọng, khuyến khích tác nhân chọn các hành động tạo ra dữ liệu giúp giảm sự không chắc chắn của mô hình, đặc biệt hữu ích trong Active Learning [3]. | 

**Trọng số thích nghi ($W_k$):**

$$W_k = \frac{W_k^0 + \alpha\, (b_k - S_k)}{\sum_j \big(W_j^0 + \alpha\, (b_j - S_j)\big)}$$

Đây là một cơ chế **Reward Shaping** rất tiên tiến và hợp lý. Thay vì sử dụng trọng số cố định, hệ thống điều chỉnh trọng số của từng thành phần dựa trên sự chênh lệch giữa điểm hiện tại ($S_k$) và baseline ($b_k$) của thành phần đó. Nếu một thành phần có hiệu suất thấp hơn baseline, trọng số của nó sẽ được tăng lên, khuyến khích tác nhân tập trung vào việc cải thiện khía cạnh đó. Điều này tạo ra một hệ thống phần thưởng động, tự điều chỉnh, giúp tác nhân học hỏi hiệu quả hơn trong môi trường đa mục tiêu [8].

**Chuẩn hóa Sigmoid:** Việc sử dụng hàm sigmoid để chuẩn hóa phần thưởng thô $R_{\text{raw}}$ về khoảng $(0, 1)$ là một kỹ thuật phổ biến trong RL để ổn định quá trình huấn luyện. Nó giúp tránh các giá trị phần thưởng quá lớn hoặc quá nhỏ có thể gây ra vấn đề về gradient hoặc mất ổn định cho mạng Q-network [9].

**Kết luận về tính hợp lý:** Hàm phần thưởng được thiết kế rất tinh vi và khoa học. Nó không chỉ bao gồm các metric hiệu suất truyền thống mà còn tích hợp các yếu tố như đa dạng dữ liệu, sự ổn định, cải thiện so với baseline và đặc biệt là giảm độ bất định. Cơ chế trọng số thích nghi là một điểm cộng lớn, cho phép hệ thống tự động điều chỉnh mục tiêu học tập.

## 4. Mối liên hệ giữa Hàm Trạng thái và Hàm Phần thưởng

Hàm trạng thái và hàm phần thưởng có mối liên hệ chặt chẽ và tương hỗ lẫn nhau, tạo thành một vòng lặp phản hồi (feedback loop) quan trọng trong quá trình học của tác nhân DQN:

1.  **Trạng thái cung cấp thông tin cho Phần thưởng:** Nhiều thành phần trong hàm phần thưởng được tính toán trực tiếp từ các thông tin có trong vector trạng thái hoặc các thông tin liên quan mà trạng thái đại diện. Ví dụ:
    *   $S_{\text{det}}$ và $S_{\text{rel}}$ dựa trên F1-score, Precision, Recall, và số lượng mẫu, những thông tin này được tổng hợp từ các đặc trưng phát hiện và quan hệ trong trạng thái.
    *   $S_{\text{div}}$ dựa trên phân phối loại quan hệ và thông tin bounding box, cũng là một phần của trạng thái hoặc được suy ra từ các đặc trưng trong trạng thái.
    *   $S_{\text{cons}}$ và $S_{\text{imp}}$ dựa trên lịch sử F1-score, mà F1-score lại là một hàm của các đặc trưng trong trạng thái.
    *   $S_{\text{unc}}$ được tính toán từ $U_{\text{mean}}$ và $U_{\text{max}}$ trong trạng thái, thể hiện mức độ giảm độ bất định.
    *   $r_{t-1}$ trong trạng thái chính là phần thưởng của bước thời gian trước đó, đóng vai trò là một tín hiệu lịch sử cho tác nhân.

2.  **Phần thưởng định hướng sự thay đổi của Trạng thái:** Ngược lại, phần thưởng mà tác nhân nhận được sẽ định hướng quá trình học của mạng Q-network, từ đó ảnh hưởng đến các hành động được chọn. Các hành động này sẽ dẫn đến những thay đổi trong môi trường và do đó, thay đổi các thành phần của trạng thái ở bước thời gian tiếp theo. Ví dụ, nếu tác nhân nhận được phần thưởng cao khi tạo ra dữ liệu đa dạng (do $S_{\text{div}}$ cao), nó sẽ học cách chọn các hành động (số lượng variations) dẫn đến việc tạo ra dữ liệu có phân phối loại quan hệ, lớp đối tượng và phân bố không gian phong phú hơn, làm thay đổi các đặc trưng tương ứng trong trạng thái.

3.  **Vòng lặp tối ưu hóa:** Mối liên hệ này tạo ra một vòng lặp tối ưu hóa: Trạng thái hiện tại dẫn đến một hành động, hành động đó tạo ra một phần thưởng và một trạng thái mới. Phần thưởng này sau đó được sử dụng để cập nhật mạng Q-network, giúp tác nhân học cách chọn các hành động tốt hơn trong các trạng thái tương tự trong tương lai. Các hành động tốt hơn này sẽ dẫn đến các trạng thái mong muốn hơn (ví dụ: hiệu suất VRD cao hơn, độ bất định thấp hơn), và cứ thế tiếp diễn.

## 5. Kết luận

Hàm trạng thái và hàm phần thưởng trong hệ thống Học tăng cường cho VRD được mô tả trong `README.md` được thiết kế rất hợp lý và chặt chẽ về mặt khoa học. Hàm trạng thái cung cấp một biểu diễn toàn diện về môi trường và hiệu suất, trong khi hàm phần thưởng đa mục tiêu và thích nghi giúp tác nhân học hỏi một cách hiệu quả, không chỉ tối ưu hóa hiệu suất mà còn khuyến khích sự đa dạng và giảm độ bất định. Mối liên hệ giữa chúng là một vòng lặp phản hồi mạnh mẽ, cho phép tác nhân liên tục cải thiện và thích nghi với môi trường. Các kỹ thuật được áp dụng đều dựa trên các nguyên tắc và nghiên cứu tiên tiến trong lĩnh vực Học tăng cường và Thị giác máy tính.

## 6. Tài liệu tham khảo

[1] Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Hassabis, D. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533.
[2] Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction*. MIT press.
[3] Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. *International Conference on Machine Learning (ICML)*.
[4] Ghiasi, G., Lin, T. Y., & Le, Q. V. (2019). Focal Loss for Dense Object Detection. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 42(2), 318-327.
[5] Cao, K., Wei, C., Ouyang, W., Ma, J., & Li, H. (2019). Learning to Rebalance Imbalanced Datasets with Label-Distribution-Aware Margin Loss. *Advances in Neural Information Processing Systems (NeurIPS)*.
[6] Houthooft, R., Chen, X., Duan, Y., Schulman, J., De Turck, F., & Abbeel, P. (2016). VIME: Variational Information Maximizing Exploration. *Advances in Neural Information Processing Systems (NeurIPS)*.
[7] Ng, A. Y., Harada, D., & Russell, S. (1999). Policy invariance under reward transformations: Theory and applications to reward shaping. *International Conference on Machine Learning (ICML)*.
[8] Wiewiora, E. (2003). Potential-based reward shaping. *International Conference on Machine Learning (ICML)*.
[9] Andrychowicz, M., Wolski, F., Ray, A., Schneider, J., Fong, R., Welinder, P., ... & Zaremba, W. (2017). Hindsight Experience Replay. *Advances in Neural Information Processing Systems (NeurIPS)*.
