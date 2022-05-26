<template>
  <div class="card h-100">
    <div class="card-body">
      <h4>
        <strong>{{ title }}</strong>
      </h4>
      <p class="card-text">
        {{ text }}
      </p>
    </div>
    <div class="card-footer">
      <slot name="footer">
        <ButtonModal></ButtonModal>
      </slot>
    </div>
  </div>
</template>
<script>
import ButtonModal from "../components/ButtonModal.vue";
export default {
  data() {},
  props: ["title", "text"],
  mounted() {},
  methods: {},
  components: { ButtonModal },
};
</script>
