<?php include '../includes/header.php'; ?>
<?php include '../includes/sidebar.php'; ?>
<?php include '../includes/utils.php'; ?>

<script src="../public/assets/js/d3.min.js"></script>

<div class="container mt-4">
  <div class="card">
    <h5 class="card-header">Indicators</h5>
    <div class="card-body p-4 p-md-5">

      <!-- Selector de año -->
      <div class="year-selector-wrap">
        <span class="year-selector-label panel-pills panel-selector-title">
          <i class=" bi bi-calendar3 me-1"></i> Year — click to toggle
        </span>
        <div  class="panel-pills"  id="year-selector"></div>
      </div>

      <!-- Panel selector -->
      <div class="panel-selector">
        <div class="panel-selector-title">
          <i class="bi bi-sliders me-1"></i> Visible panels — click to toggle
        </div>
        <div class="panel-pills" id="panel-pills"></div>
      </div>

      <!-- Cards grid -->
      <div id="panels-grid">
        <div class="state-overlay" id="loading-state">
          <div class="spinner"></div>
          Loading indicators…
        </div>
      </div>

    </div>
  </div>
</div>
</div>
</div>

<!-- Tooltip global -->
<div class="d3-tooltip" id="tooltip"></div>

<script src="./app.js"></script>

<?php include '../includes/footer.php'; ?>
