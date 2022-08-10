<template>
  <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
    <div class="container px-5">
      <router-link to="/" class="navbar-brand">Paris Baguette</router-link>
      <button
        class="navbar-toggler"
        type="button"
        data-bs-toggle="collapse"
        data-bs-target="#navbarSupportedContent"
        aria-controls="navbarSupportedContent"
        aria-expanded="false"
        aria-label="Toggle navigation"
      >
        <span class="navbar-toggler-icon"></span>
      </button>
      <div id="navbarSupportedContent" class="collapse navbar-collapse">
        <ul class="navbar-nav ms-auto mb-2 mb-lg-0">
          <li class="nav-item">
            <router-link class="nav-link" to="/upload">Upload</router-link>
          </li>
          <li class="nav-item">
            <router-link class="nav-link" to="/modify">Modify</router-link>
          </li>
          <li class="nav-item">
            <router-link class="nav-link" to="/result">Result</router-link>
          </li>
        </ul>
      </div>
    </div>
  </nav>
</template>
<script>
export default {
  name: "ComponentHeader",
};
</script>
