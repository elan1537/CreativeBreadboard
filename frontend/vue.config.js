const { defineConfig } = require('@vue/cli-service');
const path = require("path");

module.exports = defineConfig({
  transpileDependencies: true
})

module.exports = {
  chainWebpack: (config) => {
    config.resolve.alias.set('@', path.resolve(__dirname, './'));
  },
};

module.exports = {
  devServer: {
        proxy : 'http://137.184.95.69:3000'
      }
}
