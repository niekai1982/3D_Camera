## decode pattern

```c++
row_light[w][0] = (Lg>0?static_cast<usigned>(Ld):Lmax)
```
- set_size:两个channel，一个横向编码，一个纵向编码
- pattern_offset:根据total_bits计算offset？？

load every image pair and compute the maximum&minimum&bit code

loop 1:

set = 0
current=0

set_size[set] = 1

>
total_images: 44
total_patterns: 21
total_bits: 10
COUNT: 42
t: 2
bit: 9
set_idx 1
channel 0
t: 4
bit: 8
set_idx 1
channel 0
t: 6
bit: 7
set_idx 1
channel 0
t: 8
bit: 6
set_idx 1
channel 0
t: 10
bit: 5
set_idx 1
channel 0
t: 12
bit: 4
set_idx 1
channel 0
t: 14
bit: 3
set_idx 1
channel 0
t: 16
bit: 2
set_idx 1
channel 0
t: 18
bit: 1
set_idx 1
channel 0
t: 20
bit: 0
set_idx 1
channel 0
t: 22
bit: 9
set_idx 2
channel 1
t: 24
bit: 8
set_idx 2
channel 1
t: 26
bit: 7
set_idx 2
channel 1
t: 28
bit: 6
set_idx 2
channel 1
t: 30
bit: 5
set_idx 2
channel 1
t: 32
bit: 4
set_idx 2
channel 1
t: 34
bit: 3
set_idx 2
channel 1
t: 36
bit: 2
set_idx 2
channel 1
t: 38
bit: 1
set_idx 2
channel 1
t: 40
bit: 0
set_idx 2
channel 1

#### decode_pattern

- total_image
- total_patterns
- total_bits

##### get pair_wise gray_image(code and ! code)
> channel: 0, 1 col code and row code
> set: 1, 2 code or !code
> PIXEL_UNCERTAIN NaN

- define pattern image
- define min_max_image
- pixel_wise operation
  - value1 = gray_image1[i,j], value2 = gray_image[i,j]
  - min_max_image
    - min_max[0](min_value) = min(value1, value2)
    - min_max[1](max_value) = max(value1, value2)
  - use robust or not
    - not 
      - value1 > value2, pattern[channel] += 1 <<bit
    - use robust
      - L = row_light, L[0] = Ld(direct_light), L[1] = Lg(glob_light)
      - p = get_robust_bit(value1, value2, L[0], L[1], m)
        - get_robust_bit
          - Ld < m BIT_UNVERTAIN, 
          - Ld > Lg value1 > value2 ?1 :0
- 

