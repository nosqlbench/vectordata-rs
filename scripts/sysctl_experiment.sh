  sudo tee /etc/sysctl.d/99-veks-writeback.conf <<'EOF'                                                                                                                            
  vm.dirty_background_bytes = 1073741824                                                                                                                                           
  vm.dirty_bytes = 4294967296                                                                                                                                                      
  vm.dirty_writeback_centisecs = 500                                                                                                                                               
  vm.dirty_expire_centisecs = 3000                                                                                                                                                 
  EOF                                                                      
  sudo sysctl --system                                
 
