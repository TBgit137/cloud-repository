happy(dog).

%如果狗开心，那么就会玩水
%结果 :- 前置条件
play(water) :- happy(dog).

unhappy(dog).
%狗不开心就会狗叫
play(bark) :- unhappy(dog).