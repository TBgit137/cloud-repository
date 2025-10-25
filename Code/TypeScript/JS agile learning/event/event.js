var btn = document.getElementById("btn")

// DOM0
btn.onclick = function(){
    console.log("clicked");
}

// DOM2
btn.addEventListener("click", function(){
    console.log("clicked");
})