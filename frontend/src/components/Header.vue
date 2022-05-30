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
      <div class="collapse navbar-collapse" id="navbarSupportedContent">
        <ul class="navbar-nav ms-auto mb-2 mb-lg-0">
          <li class="nav-item">
            <router-link class="nav-link" to="/upload">Upload</router-link>
          </li>
          <li v-if="isShowing" class="nav-item">
            <router-link class="nav-link" to="/modify">Modify</router-link>
          </li>
          <li v-if="isShowing" class="nav-item">
            <router-link class="nav-link" to="/result">Result</router-link>
          </li>
        </ul>
      </div>
    </div>
  </nav>
</template>
<script>
export default {
  data() {
    return {
      isShowing: false,
    };
  },
  created() {
    let img = localStorage.img;
    // let origin_img = localStorage.origin_img;
    // let circuit_img = localStorage.circuit_img;

    if (img) {
      this.isShowing = true;
    } else {
      this.isShowing = false;
    }
  },
};
</script>
