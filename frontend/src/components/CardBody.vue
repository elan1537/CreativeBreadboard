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
      <slot name="footer"> </slot>
    </div>
  </div>
</template>
<script>
export default {
  name: "ComponentCardBody",
  props: {
    title: {
      type: String,
      required: true,
    },
    text: {
      type: String,
      required: true,
    },
  },
};
</script>
