var arr1 = ["cyka", "blyat", "nahui", "mudak", "yebat"];

var arr2 = ["shit", "fuck", "ass", "bitch", "cunt"];

console.log(Array.isArray(arr1)); //isArray() - 判断是否为数组

console.log(arr1.push("shit")); //push() - 向数组末尾添加一个或多个元素，返回新的数组长度,在原数组上修改

console.log(arr1.pop()); //pop() - 删除数组末尾的元素，返回被删除的元素,在原数组上修改

console.log(arr1.shift()); //shift() - 删除数组开头的元素，返回被删除的元素,在原数组上修改

console.log(arr1.unshift("shit")); //unshift() - 向数组开头添加一个或多个元素，返回新的数组长度,在原数组上修改

console.log(arr1.join()); //join() - 将数组转换为字符串，用括号内参数分割，返回新的字符串,没有参数则用逗号连接,在原数组上修改

console.log(arr1.concat(arr2)); //concat() - 连接两个或更多数组，返回一个新的数组,在原数组上修改,作为参数的数组不会被修改

console.log(arr1.reverse()); //reverse() - 反转数组，返回新的数组,在原数组上修改

console.log(arr1.indexOf("blyat")); //indexOf() - 返回指定元素的索引,如果没有找到则返回-1


