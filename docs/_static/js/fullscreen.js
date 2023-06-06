// fullscreen.js
document.addEventListener("DOMContentLoaded", function() {
    var fullscreenImages = document.getElementsByClassName("fullscreen-image");

    for (var i = 0; i < fullscreenImages.length; i++) {
      fullscreenImages[i].addEventListener("click", function() {
        if (!document.fullscreenElement) {
          this.requestFullscreen();
        } else {
          if (document.exitFullscreen) {
            document.exitFullscreen();
          }
        }
      });
    }
  });
