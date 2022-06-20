<template>
  <div class="container px-4 px-lg-5">
    <!-- Heading Row-->
    <div class="row gx-4 gx-lg-5 align-items-center my-5">
      <div class="col-lg-7">
        <img
          class="img-fluid rounded mb-4 mb-lg-0"
          :src="img"
          width="600"
          height="800"
          alt="..."
        />
      </div>
      <div class="col-lg-5">
        <div class="row">
          <img
            class="img-fluid rounded mb-4 mb-lg-4"
            :src="circuit"
            width="600"
            height="800"
            alt="..."
          />
        </div>
        <div class="row mb-3">
          <div class="col">
            <div class="dropdown">
              <div class="d-grid gap-2">
                <button
                  class="btn btn-secondary dropdown-toggle"
                  href="#"
                  role="button"
                  id="dropdownMenuLink"
                  data-bs-toggle="dropdown"
                  aria-expanded="false"
                >
                  전압
                </button>
                <ul class="dropdown-menu" aria-labelledby="dropdownMenuLink">
                  <li
                    v-for="(row, idx) in JSON.parse(
                      circuit_analysis['node_voltage']
                    )"
                    :key="`${row}_${idx}`"
                  >
                    <a class="dropdown-item"
                      >Node-{{ idx }}::{{ parseFloat(row[0]).toFixed(3) }}V</a
                    >
                  </li>
                </ul>
              </div>
            </div>
          </div>
          <div class="col">
            <div class="dropdown">
              <div class="d-grid gap-2">
                <button
                  class="btn btn-secondary dropdown-toggle"
                  href="#"
                  role="button"
                  id="dropdownMenuLink"
                  data-bs-toggle="dropdown"
                  aria-expanded="false"
                >
                  전류
                </button>
                <ul class="dropdown-menu" aria-labelledby="dropdownMenuLink">
                  <li>
                    {{
                      parseFloat(circuit_analysis["node_current"]).toFixed(3)
                    }}A
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>
        <div class="row">합성저항 :: {{ circuit_analysis["r_th"] }}</div>
      </div>
    </div>
  </div>
</template>
<style></style>
<script>
export default {
  name: "Result",
  data() {
    return {
      img: "data:image/png;base64,",
      circuit: "data:image/png;base64,",
      circuit_analysis: null,
      detected_components: null,
    };
  },
  created() {
    if (localStorage.img) {
      this.img += localStorage.img;
      this.circuit += localStorage.circuit_img;
      this.circuit_analysis = JSON.parse(localStorage.circuit_analysis);
      this.detected_components = JSON.parse(localStorage.detected_components);
    }
  },
};
</script>
