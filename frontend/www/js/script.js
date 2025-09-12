// Espera a que todo el HTML de la página esté cargado antes de ejecutar el script
document.addEventListener('DOMContentLoaded', function () {

    // Selecciona todos los enlaces de navegación de la barra lateral
    const navLinks = document.querySelectorAll('.nav-link');
    
    // Selecciona el contenedor donde se mostrará el contenido
    const contentContainer = document.getElementById('contenido-principal');

    // Función para cargar el contenido de una página
    function loadContent(page) {
        // Muestra un estado de carga mientras se obtiene el contenido
        contentContainer.innerHTML = '<div class="d-flex justify-content-center"><div class="spinner-border" role="status"><span class="visually-hidden">Cargando...</span></div></div>';

        // Usa la función fetch para obtener el contenido del archivo PHP de forma asíncrona
        fetch(page)
            .then(response => {
                // Comprueba si la respuesta del servidor es correcta
                if (!response.ok) {
                    throw new Error('La respuesta de la red no fue correcta');
                }
                // Convierte la respuesta a texto (que será nuestro HTML)
                return response.text();
            })
            .then(html => {
                // Inserta el HTML recibido dentro de nuestro contenedor principal
                contentContainer.innerHTML = html;
            })
            .catch(error => {
                // Si hay un error, lo muestra en la consola y en la página
                console.error('Error al cargar la página:', error);
                contentContainer.innerHTML = `<div class="alert alert-danger">Error al cargar el contenido. Por favor, intenta de nuevo.</div>`;
            });
    }

    // Añade un evento 'click' a cada uno de los enlaces del menú
    navLinks.forEach(link => {
        link.addEventListener('click', function (event) {
            // Previene el comportamiento por defecto del enlace (que es recargar la página)
            event.preventDefault();

            // Quita la clase 'active' de todos los enlaces
            navLinks.forEach(l => l.classList.remove('active'));
            // Añade la clase 'active' solo al enlace que fue clickeado
            this.classList.add('active');

            // Obtiene el nombre del archivo a cargar desde el atributo 'data-page'
            const pageToLoad = this.getAttribute('data-page');
            
            // Llama a la función para cargar el contenido
            loadContent(pageToLoad);
        });
    });
});
