### calibrate

#### load pattern images

??threshold shadow_threshold????
read shadow_threshold = 10
corner_count_x = 7
corner_count_y = 11
corners_width = 21
corners_height = 21.08

---
model_rowCount ??? images_num? or image_set number????
level meaning???
image_set num
---

pattern_set[pattern_list, pattern_list, pattern_list]

std::vector<vector<point2f>> chessboard_corners 
vector<vector<point2f>> projector_corners 
vector<cv::Mat> pattern_list 


pattern_set -> pattern_list
step1:
decode


