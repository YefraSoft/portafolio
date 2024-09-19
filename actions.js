const darkModeButton = document.getElementById("slideButton");
isActive = false;

darkModeButton.addEventListener("click", function () {
  if (isActive) {
    isActive = false;
    document.body.classList.toggle("dark-mode");
    document.body.classList.toggle("body-active");
    darkModeButton.classList.toggle("togle-button-isActive");
  } else {
    isActive = true;
    document.body.classList.toggle("dark-mode");
    document.body.classList.toggle("body-active");
    darkModeButton.classList.toggle("togle-button-isActive");
  }
});

document.addEventListener('DOMContentLoaded', function() {
  // Seleccionar todas las barras de progreso
  const progressBars = document.querySelectorAll('.prog-bar');

  // Iterar sobre cada barra y actualizar su ancho según el atributo data-progress
  progressBars.forEach(bar => {
    const progress = bar.querySelector('.prog');
    const percentage = bar.getAttribute('data-progress');
    
    // Actualizar el ancho de la barra interna según el porcentaje
    progress.style.width = percentage + '%';
  });
});