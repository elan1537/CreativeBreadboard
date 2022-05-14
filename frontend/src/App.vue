<template>
  <div>
    <Header></Header>
    <div class="container px-4 px-lg-5">
      <ImageModify></ImageModify>
    </div>
    <Footer></Footer>
  </div>
</template>

<script>
// import HelloWorld from "./components/HelloWorld.vue";
// import ImageViewer from "./components/ImageViewer.vue";
import Header from "./components/Header.vue";
import ImageModify from "./components/ImageModify.vue";
import Footer from "./components/Footer.vue";

export default {
  name: "App",
  components: {
    // ImageViewer,
    // HelloWorld,
    ImageModify,
    Header,
    Footer,
  },
};
</script>

<style></style>
