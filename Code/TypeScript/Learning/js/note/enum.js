"use strict";
// 目的：限制数据值范围，number，string这样的数据类型限制太过宽泛
Object.defineProperty(exports, "__esModule", { value: true });
// 数字枚举
var Week;
(function (Week) {
    Week[Week["Monday"] = 1] = "Monday";
    Week[Week["Tuesday"] = 2] = "Tuesday";
    Week[Week["Wednesday"] = 3] = "Wednesday";
    Week[Week["Thursday"] = 4] = "Thursday";
    Week[Week["Friday"] = 5] = "Friday";
    Week[Week["Saturday"] = 6] = "Saturday";
    Week[Week["Sunday"] = 7] = "Sunday";
})(Week || (Week = {}));
console.log(Week[1]); // Monday可以由值取key，也可以由key取值
var StringWeek;
(function (StringWeek) {
    StringWeek["Monday"] = "theMonday";
    StringWeek["Tuesday"] = "Tuesday";
    StringWeek["Wednesday"] = "Wednesday";
    StringWeek["Thursday"] = "Thursday";
    StringWeek["Friday"] = "Friday";
    StringWeek["Saturday"] = "Saturday";
    StringWeek["Sunday"] = "Sunday";
})(StringWeek || (StringWeek = {}));
console.log(StringWeek.Monday); // √
console.log(StringWeek["Monday"]); // √
// × 不可以由值取key，只能由key到值
