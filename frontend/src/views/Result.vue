<template>
  <div class="container px-4 px-lg-5">
    <!-- Heading Row-->
    <div class="row gx-4 gx-lg-5 align-items-center my-5">
      <div class="col-lg-7">
        <img
          class="img-fluid rounded mb-4 mb-lg-0"
          :src="img_url"
          width="600"
          height="800"
          alt="..."
        />
      </div>
      <div class="col-lg-5">
        <div class="row">
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
                  <li><a class="dropdown-item" href="#">젠장</a></li>
                  <li><a class="dropdown-item" href="#">하나도</a></li>
                  <li><a class="dropdown-item" href="#">모르겠어</a></li>
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
                  <li><a class="dropdown-item" href="#">전문가를</a></li>
                  <li><a class="dropdown-item" href="#">모셔서</a></li>
                  <li><a class="dropdown-item" href="#">합시다</a></li>
                </ul>
              </div>
            </div>
          </div>
        </div>
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
      img_url: require("../../../backend/static/uploads/Circuit-7.220428.jpg"),
    };
  },
};
</script>
