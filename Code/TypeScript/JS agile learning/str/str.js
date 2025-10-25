var str1 = "cykablyat";
var str2 = "nahui";
var str3 = "    cykablyat    ";

console.log(str1.charAt(0)); //charAt() - 返回指定索引位置的字符

console.log(str1.concat(str2)); //concat() - 连接两个或更多字符串，返回一个新的字符串

console.log(str1.substring(0, 4)); //substring() - 返回指定起止索引之间的子字符串

console.log(str1.substr(4, 5)); //substr() - 从指定位置开始，返回指定长度的子字符串

console.log(str1.indexOf("a", 4)); //indexOf() - 返回指定字符串首次出现的索引，从指定位置开始搜索

console.log(str3.trim()); //trim() - 去除字符串两端的空白字符，还可以去掉\t,\v,\r,\n等空白字符

console.log(str1.split("a")); //split() - 将字符串以指定字符分割为字符串数组，返回一个新数组，第二个参数为数组的长度