<template>
  <div>
    <Header></Header>
    <router-view></router-view>
    <Footer></Footer>
  </div>
</template>

<script>
// import HelloWorld from "./components/HelloWorld.vue";
// import ImageViewer from "./components/ImageViewer.vue";
import Header from "./components/Header.vue";
import Footer from "./components/Footer.vue";

export default {
  name: "App",
  components: {
    // ImageViewer,
    // HelloWorld,
    Header,
    Footer,
  },
};
</script>

<style></style>
