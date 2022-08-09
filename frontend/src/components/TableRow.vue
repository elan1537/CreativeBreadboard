<template>
  <tr>
    <td scope="col"></td>
    <td>{{ value["name"] }}</td>

    <td>
      <select class="form-select" aria-label="클래스">
        <option selected value="1">Line</option>
        <option value="2">Resistor</option>
      </select>
    </td>

    <td v-if="click_table[value['name']]">
      <input
        v-model="detected_line_components[value['name']]['start']"
        type="text"
        class="form-control"
        placeholder="시작점"
        aria-label="시작점"
      />
    </td>

    <td v-if="click_table[value['name']]">
      <input
        v-model="detected_line_components[value['name']]['end']"
        type="text"
        class="form-control"
        placeholder="끝점"
        aria-label="끝점"
      />
    </td>

    <td v-if="click_table[value['name']]">
      <div @click="edit_row(value['name'])">update</div>
      <div @click="click_table[value['name']] = false">cancel</div>
    </td>

    <td v-else @click="edit_row(value['name'])">edit</td>
  </tr>
</template>
<script>
export default {
  name: "ComponentTableRow",
  props: [],
  data() {
    return {};
  },
};
</script>
