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


