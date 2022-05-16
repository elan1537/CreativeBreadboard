<template>
  <div class="container px-4 px-lg-5">
    <h1>Modify View</h1>
    <div>
      <form action="/image" method="POST" enctype="multipart/form-data">
        <input type="submit" onsubmit="refresh" />
      </form>
    </div>
    <ImageModify></ImageModify>
  </div>
</template>
<script>
import ImageModify from "../components/ImageModify.vue";

export default {
  components: { ImageModify },
  data() {
    return {
      image: "",
    };
  },
  methods: {
    uploadImg() {
      console.log("들어왔다");
      var image = this.$refs["image"].files[0];

      const url = URL.createObjectURL(image);
      this.image = url;
      console.log(url);
      console.log(this.image);
    },
  },
};
</script>
