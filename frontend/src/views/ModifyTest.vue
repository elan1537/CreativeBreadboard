<template>
  <div class="container px-4 px-lg-5">
    <h1>Modify </h1>
    <div>
      <form action="/image" method="POST" enctype="multipart/form-data">
        <input type="submit" onsubmit="refresh" />
      </form>
    </div>
  </div>
</template>
<script>

export default {
  components: {  },
  data() {
    return {
      image: "",
    };
  },
  methods: {
    uploadImg() {
      var image = this.$refs["image"].files[0];

      const url = URL.createObjectURL(image);
      this.image = url;
    },
  },
}
</script>
