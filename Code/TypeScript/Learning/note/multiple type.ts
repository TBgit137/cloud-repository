// union type
let data:string | number = "llm";

data = 12;

// intersection type
type Obj1={name:string};
let obj1:Obj1={name:"Larry"};

type Obj2={age:number};
let obj2:Obj2={age:13};

let obj3:Obj1 & Obj2={name:"Steve", age:12}

// literal type
type num = 1|2|3;
let n:num = 1;