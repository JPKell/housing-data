function parallax() {
	var s = document.getElementById("floater");
  var yPos = 0 - window.pageYOffset*4;	
  s.style.top = yPos + "%";
}

window.addEventListener("scroll", function(){
	parallax();	
});