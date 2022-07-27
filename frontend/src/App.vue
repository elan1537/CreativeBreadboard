<script>
import Header from "./components/Header.vue";
import Footer from "./components/Footer.vue";

// eslint-disable-next-line vue/no-export-in-script-setup
export default {
  name: "App",
  components: {
    Header,
    Footer,
  },
};
</script>

<template>
  <Header />
  <router-view />
  <Footer />
</template>

<style scoped>

</style>
