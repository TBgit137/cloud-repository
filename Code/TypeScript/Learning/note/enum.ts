// 目的：限制数据值范围，number，string这样的数据类型限制太过宽泛

// 数字枚举
enum Week{
    Monday = 1, //对应的值会类推下去（Tuesday = 2）
    Tuesday,
    Wednesday,
    Thursday,
    Friday,
    Saturday,
    Sunday
}
console.log(Week[1]); // Monday可以由值取key，也可以由key取值

enum StringWeek{
    Monday = "theMonday",
    Tuesday = "Tuesday",
    Wednesday = "Wednesday",
    Thursday = "Thursday",
    Friday = "Friday",
    Saturday = "Saturday",
    Sunday = "Sunday"
}

console.log(StringWeek.Monday); // √
console.log(StringWeek["Monday"]); // √
// × 不可以由值取key，只能由key到值
