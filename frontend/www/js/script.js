    document.addEventListener('DOMContentLoaded', function() {
        
        const sidebarToggle = document.getElementById('sidebar-toggle');
        const wrapper = document.querySelector('.wrapper');
        const toggleIcon = sidebarToggle.querySelector('.icon');
        const toggleText = sidebarToggle.querySelector('.text');

        // FUNCIÓN PARA ACTUALIZAR EL BOTÓN (REUTILIZABLE)
        function updateToggleButton() {
            if (wrapper.classList.contains('collapsed')) {
                // Si está encogido
                toggleIcon.classList.remove('bi-chevron-left');
                toggleIcon.classList.add('bi-chevron-right');
                //toggleText.textContent = ''; // Ocultamos texto para un mejor centrado
            } else {
                // Si está expandido
                toggleIcon.classList.remove('bi-chevron-right');
                toggleIcon.classList.add('bi-chevron-left');
                //toggleText.textContent = 'Encoger';
            }
        }

        // Evento de clic para el botón
        sidebarToggle.addEventListener('click', function() {
            wrapper.classList.toggle('collapsed');
            // Actualizamos el botón cada vez que se hace clic
            updateToggleButton();
        });

        // --- CÓDIGO NUEVO ---
        // COMPRUEBA EL ESTADO INICIAL AL CARGAR LA PÁGINA
        function checkInitialSidebarState() {
            // El punto de ruptura 'md' de Bootstrap es 768px
            if (window.innerWidth < 768) {
                wrapper.classList.add('collapsed');
            }
            // Actualizamos el botón para que refleje el estado inicial
            updateToggleButton();
        }
        
        // Ejecutamos la comprobación inicial
        checkInitialSidebarState();

    });
