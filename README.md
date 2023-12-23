# **1. Tìm hiểu, so sánh các phương pháp Optimizer trong huấn luyện mô hình học máy**
## **1.1	Tìm hiểu về Optimizer** 
Trước khi ta đi sâu vào tìm hiểu Optimizer và các thuật toán trong nó, ta cần hiểu thế nào là thuật toán tối ưu (optimizer). Hiểu một cách đơn giản và cụ thể, thuật toán tối ưu là cơ sở xây dựng mô hình Neutral Network, nhằm “học” được các đặc điểm hoặc mẫu (feature hoặc pattern) của dữ liệu đầu vào. Từ đó, mục tiêu là tìm ra một cặp trọng số (weights) và độ lệch (bias) phù hợp để tối ưu hóa mô hình. Nhưng khó khăn ở đây là làm thế nào để có thể tìm ra các trọng số và độ lệch phù hợp để tránh lãng phí tài nguyên. Và đó là lý do vì sao các thuật toán tối ưu ra được ra đời.

## **1.2	Các thuật toán tối ưu**
### **1.2.1 Gradient Descent**
Trong việc tối ưu hóa mạng Neutral, Gradient Descent (GD) là một trong những thuật toán phổ biến nhất. Được thiết kế nhầm mục đích giảm thiểu hàm mất mát (loss function) J (θ), trong đó (θ) đại diện cho tập hợp các trọng số (weights) của mô hình cần được tối ưu. Quy tắc của GD được tổng quát:θ_(t+1)= θ_t- η.∇_θ J(θ_t )
Trong đó, ∇_θ J(θ_t ) biểu thị gradient của hàm mất mát tại θ ở bước t. η là một giá trị dương được gọi là tốc độ học (learning rate), quyết định kích thước của các bước di chuyển đến giá trị cực tiểu (hoặc cực tiểu địa phương) gọi là local minimum.

### **1.2.2 Batch Gradient Descent**
Batch Gradient Desscent dùng để tính gradient của hàm mất mát tại θ trên toàn bộ tập dữ liệu. Mọi điểm dữ liệu đều được sử dụng để tính gradient trước khi cập nhật bộ trọng số θ. Tuy nhiên, Batch GD có hạn chế khi xử lý tập dữ liệu lớn vì đòi hỏi nhiều thời gian và chi phí tính toán.

### **1.2.3 Stochastic Gradient Descent (SGD)**
Để khác phục hạn chế của Batch Gradient Descent, thuật toán Stochastic Gradient Descent ra đời để thực hiện cập nhật trọng số sau mỗi mẫu dữ liệu x^((ⅈ) ) có nhãn tương ứng y^((ⅈ) ) như sau:
θ_(t+1)= θ_t- η.∇_θ J(θ_t;x^((ⅈ) )  ;y^((ⅈ) ) )
Với cách cập nhật này, SGD thường nhanh hơn Batch GD và có thể áp dụng vào quá trình học trực tuyến (online learning) khi tập huấn luyện được cập nhật liên tục dữ liệu mới. Trong SGD bộ trọng số θ thường được cập nhật liên tục hơn Batch GD, chính vì vậy mà hàm mất mát dao động nhiều hơn. Nhưng điều này lại gây khó khăn cho SGD có vẻ không ổn định, nhưng điểm đặc biệt là sự di chuyển của các điểm locol minimum có tiềm năng lớn hơn. Đồng thời, tốc độ học (learning rate) giảm, khả năng hội tụ của SGD cũng tương đương với Batch GD.

### **1.2.4 Mini-batch Gradient Descent**
Mini-batch Gradient Descent khá khác với các thuật toán trước đó, Mini-batch GD sử dụng k điểm dữ liệu để cập nhật trọng số (1 < k < N với N là tổng số điểm dữ liệu).
θ_(t+1)= θ_t- η.∇_θ J(θ_t;x^((ⅈ:i+k) )  ;y^((ⅈ:i+k) ) )
Mini-batch GD giảm độ biến động của hàm mất mát so với SGD và chi phí tính toán gradient với k điểm dữ liệu là chấp nhận được. Khi huấn luyện mạng Neutral, Mini-batch GD thường được ưu tiên chọn, và do đó, trong một số trường hợp, nó được coi là một biến thể của SGD. Tuy nhiên, Mini-batch GD một mình không đảm bảo việc đạt được điểm cực tiểu của hàm mất mát, và các yếu tố như tốc độ học, đặc tính của dữ liệu, và đặc điểm của hàm mất mát cũng đóng vai trò quan trọng trong quá trình này.

### **1.2.6 Gradient Descent và các biến thể**
| Thuật toán      | Dữ liệu sử dụng  | Gradients               |
|-----------------|-----------------|--------------------------------|
| GD              | Mọi dữ liệu     | 1/n ∑_(i=1)^n▒∇f(z_i )         |                        
| SGD             | Dữ liệu đơn lẻ   | ∇f(z_i )                       |                       
| Batch GD        | Hàng loạt dữ liệu| 1/m ∑_(i=1)^m▒∇f(z_i ) ,m<n    |                        
| Mini-batch GD   | Hàng loạt nhỏ   | 1/m ∑_(i=1)^m▒∇f(z_i ) ,m≪n   |                        

### **1.2.7 Thử thách với SGD**
•	Lựa chọn một learning rate phù hợp là một nhiệm vụ rất khó. 
•	Một lịch trình learning rate duy nhất có thể không thích ứng được với các bộ dữ liệu đa dạng. 
•	Áp dụng cùng một learning rate cho tất cả các tham số có thể không phải là lựa chọn tốt nhất. 
•	Object function cho mạng Neutral có tính phi lồi cao, đồng nghĩa với việc có nhiều điểm cực tiểu địa phương. 

### **1.2.8 Momemtum**
Để khắc phục được những hạn chế trên của thuật toán Gradient Descent, ta sẽ dùng Gradient Descent với Momemtum. Dưới đây là ví dụ về GD với Momemtum:
<img src="https://github.com/proplayer-team/51900836_Final_MachineLearning/blob/main/q%C6%B0.jpg">
Nhìn vào hình bên trên, ta thả hai viên bi từ hai điểm khác nhau A và B, viên bi ở A sẽ trượt xuống đến điểm C, trong khi viên bi ở B sẽ trượt xuống điểm D. Tuy nhiên, chúng ta không muốn viên bi ở B dừng lại ở điểm D (địa phương tối thiểu), mà thay vào đó, nó sẽ tiếp tục lăn xuống điểm C (địa phương tối thiểu toàn cục). Để thực hiện điều này, chúng ta cần cung cấp cho viên bi ở B một vận tốc ban đầu đủ lớn để nó có thể vượt qua điểm E và đến điểm C. Dựa trên ý tưởng này, thuật toán Momentum được phát triển (hay còn gọi là theo đà tiến tới). Tính toán học của thuật toán Momentum có công thức như sau:
xnew = xold - (gama.v + learningrate.gradient)
Trong đó:
•	xnew: tọa độ mới 
•	xold: tọa độ cũ 
•	gama: parameter, thường =0.9 
•	learningrate: tốc độ học 
•	gradient: đạo hàm của hàm f
Qua ví dụ trên, ta thấy viên bi sẽ vượt tốc tiến tới điểm global minimum và dao động qua lại quanh điểm đó trước khai dừng lại. Đó cũng chính là ưu điểm của thuật toán so với Gradient Descent thông thường bằng việc tiến được đến điểm global minimum và không chỉ dừng lại ở local minimum.

