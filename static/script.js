// Variable para guardar la gráfica y poder borrarla si suben otra imagen
let grafica = null;
let nivelZoom = 100; // Variable para controlar el nivel de zoom de la imagen

// Previsualizar imagen apenas se selecciona
document.getElementById('image-input').addEventListener('change', function (event) {
    const imagen = event.target.files[0];
    const preview = document.getElementById('preview');
    const wrapper = document.getElementById('image-wrapper');
    const controlesZoom = document.getElementById('zoom-controls');

    if (imagen) {
        const reader = new FileReader();
        reader.onload = function (e) {
            preview.src = e.target.result;
            wrapper.style.display = 'block'; // Muestra el contenedor
            controlesZoom.style.display = 'block'; // Mostrar botones de zoom

            // Reiniciar el zoom cada que se sube una foto nueva
            nivelZoom = 100;
            preview.style.width = `${nivelZoom}%`;
        };
        reader.readAsDataURL(imagen);
    } else {
        preview.src = "";
        wrapper.style.display = 'none';
        controlesZoom.style.display = 'none'; // Ocultar botones de zoom
    }
});

// Botones de zoom in y zoom out
document.getElementById('zoom-in').addEventListener('click', function () {
    nivelZoom += 30; // Aumenta el width 30%
    document.getElementById('preview').style.width = `${nivelZoom}%`;
});

document.getElementById('zoom-out').addEventListener('click', function () {
    // Evitamos que se haga demasiado pequeña
    if (nivelZoom > 100) {
        nivelZoom -= 30; // Reduce 30%
        document.getElementById('preview').style.width = `${nivelZoom}%`;
    }
});

document.getElementById('upload-form').addEventListener('submit', function (event) {
    // Evita que la página se recargue
    event.preventDefault();

    // Obtenemos la imagen que el usuario seleccionó
    const inputImagen = document.getElementById('image-input');
    const imagen = inputImagen.files[0];
    // Preparamos la imagen para enviarla a la API
    const datosEnvio = new FormData();
    datosEnvio.append('image', imagen);

    fetch('/predict', {
        method: 'POST',
        body: datosEnvio
    })
        .then(respuesta => respuesta.json()) // Convertimos la respuesta a texto JSON
        .then(datos => {
            // Si la API regresa un error
            if (datos.error) {
                document.getElementById('error-message').innerText = datos.error;
                document.getElementById('error-message').style.display = 'block';
                document.getElementById('results-section').style.display = 'none';
            } else {
                // Si salió bien ocultamos errores y mostramos resultados
                document.getElementById('error-message').style.display = 'none';
                document.getElementById('results-section').style.display = 'block';

                // Mostramos los resultados en la página
                document.getElementById('pred-class').innerText = datos.prediccion;
                document.getElementById('pred-prob').innerText = datos.probabilidad;

                // Dibujamos la gráfica
                dibujarGrafica(datos.detalles);
            }
        });
});

// Función para dibujar la gráfica con Chart.js
function dibujarGrafica(detalles) {
    const lienzo = document.getElementById('probabilityChart').getContext('2d');

    // Convertimos los porcentajes de texto a decimales
    const probBenigno = parseFloat(detalles.probabilidad_benigno);
    const probMaligno = parseFloat(detalles.probabilidad_maligno);

    // Si ya existía una gráfica de antes, se elimina
    if (grafica != null) {
        grafica.destroy();
    }

    // Creamos la gráfica
    grafica = new Chart(lienzo, {
        type: 'bar',
        data: {
            labels: ['Benigno', 'Maligno'],
            datasets: [{
                label: 'Probabilidad (%)',
                data: [probBenigno, probMaligno],
                backgroundColor: ['#38a169', '#e53e3e']
            }]
        },
        options: {
            maintainAspectRatio: false,
            scales: {
                y: { max: 100 }
            }
        }
    });
}
