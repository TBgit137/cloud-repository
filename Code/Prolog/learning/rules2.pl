happy(zhangsan).
listenmusic(zhangsan).
listenmusic(lisi).

%如果李四听音乐，他就会开心
happy(lisi) :- listenmusic(lisi).

%如果张三快乐，他就会弹吉他
# playguitar(zhangsan) :- happy(zhangsan).

%如果张三开心并且在听音乐，那么他就弹奏吉他
playguitar(zhangsan) :- happy(zhangsan),listenmusic(zhangsan).

%狗如果吃肉或者出去玩就很开心
eat(cat).
play(dog).
happy(dog) :- eat(dog) ; play(dog).