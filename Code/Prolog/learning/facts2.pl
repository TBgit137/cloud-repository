man(zhangsan).
man(lisi).
man(wangwu).

love(zhangsan, lisi).
love(wangwu, lisi).
%如果A爱上B，C也爱上B，那么A会嫉妒C
jealous(A,C) :- love(A,B),love(C,B).