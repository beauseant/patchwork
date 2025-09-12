<!-- Sidebar fija en pantallas medianas o mayores -->
<aside class="col-md-3 col-lg-2 bg-light border-end vh-100 d-none d-md-block">
    <nav class="nav flex-column p-3 position-sticky" style="top: 0;">
        <a class="nav-link" href="load.php">Subir documento</a>
        <a class="nav-link" href="list.php">Ver estado</a>
        <a class="nav-link" href="tres.php">Sección 3</a>
    </nav>
</aside>

<!-- Sidebar tipo offcanvas en móviles (izquierda) -->
<div class="offcanvas offcanvas-start d-md-none" tabindex="-1" id="sidebarMenu">
    <div class="offcanvas-header">
        <h5 class="offcanvas-title">Menú</h5>
        <button type="button" class="btn-close" data-bs-dismiss="offcanvas"></button>
    </div>
    <div class="offcanvas-body">
        <nav class="nav flex-column">
            <a class="nav-link" href="load.php">Subir documento</a>
            <a class="nav-link" href="list.php">Ver estado</a>
            <a class="nav-link" href="tres.php">Sección 3</a>
        </nav>
    </div>
</div>
